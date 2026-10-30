"""Hardware profile detection for LinguaBee.

Detects the current machine's capabilities and returns a profile string
used to set intelligent defaults for model backends and devices.

Profiles:
  "apple_silicon"  – macOS arm64 with Apple Silicon (M1/M2/M3/M4)
  "nvidia_gpu"     – CUDA-capable NVIDIA GPU with >= 8 GB VRAM
  "cpu"            – fallback (Intel Mac, Windows/Linux CPU-only)

Usage:
    from hardware_profile import get_profile, get_auto_defaults
    profile = get_profile()
    defaults = get_auto_defaults()
"""
import platform
import sys
import os

# Cache so detection only runs once per process.
_cached_profile: str | None = None
_cached_gpu_name: str | None = None
_cached_vram_gb: float = 0.0


def _detect() -> tuple[str, str, float]:
    """Return (profile, gpu_name, vram_gb)."""
    system = platform.system()
    machine = platform.machine()

    # --- Apple Silicon ---
    if system == "Darwin" and machine == "arm64":
        # Check if mlx is available (best backend for Apple Silicon)
        try:
            import mlx.core  # noqa: F401
            mlx_available = True
        except ImportError:
            mlx_available = False

        gpu_name = f"Apple {platform.processor() or 'Silicon'}"

        # Unified memory – report physical RAM as "VRAM" equivalent
        try:
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL
            ).decode().strip()
            vram_gb = int(out) / (1024 ** 3)
        except Exception:
            vram_gb = 0.0

        return "apple_silicon", gpu_name, vram_gb

    # --- NVIDIA CUDA or AMD ROCm ---
    # PyTorch built with ROCm/HIP also reports torch.cuda.is_available() == True
    # and uses the same "cuda" device string.  The presence of torch.version.hip
    # (a non-None string like "6.2.4") distinguishes ROCm from CUDA.
    try:
        import torch
        if torch.cuda.is_available():
            idx = 0
            try:
                props = torch.cuda.get_device_properties(idx)
                vram_gb = props.total_memory / (1024 ** 3)
                gpu_name = props.name
            except Exception:
                vram_gb = 0.0
                gpu_name = "GPU"

            hip_version = getattr(torch.version, "hip", None)
            if hip_version is not None:
                # AMD ROCm/HIP backend – use a lower VRAM threshold because
                # integrated AMD GPUs (e.g. Ryzen AI Max+ 395) share system RAM,
                # so even 4 GB of reported VRAM is enough to run inference.
                if vram_gb >= 4.0:
                    return "amd_gpu", gpu_name, vram_gb
            else:
                # NVIDIA CUDA backend
                if vram_gb >= 8.0:
                    return "nvidia_gpu", gpu_name, vram_gb
    except ImportError:
        pass

    return "cpu", "", 0.0


def get_profile() -> str:
    """Return the hardware profile string (cached after first call)."""
    global _cached_profile, _cached_gpu_name, _cached_vram_gb
    if _cached_profile is None:
        _cached_profile, _cached_gpu_name, _cached_vram_gb = _detect()
    return _cached_profile


def get_gpu_name() -> str:
    get_profile()  # ensure cache populated
    return _cached_gpu_name or ""


def get_vram_gb() -> float:
    get_profile()  # ensure cache populated
    return _cached_vram_gb


def is_mlx_available() -> bool:
    """Return True if mlx-lm is importable (Apple Silicon backend)."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def is_llama_cpp_available() -> bool:
    """Return True if llama-cpp-python is importable."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def _ctranslate2_has_gpu() -> bool:
    """Return True if the installed CTranslate2 build can use a GPU (CUDA or ROCm)."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def get_auto_defaults() -> dict:
    """Return a dict of recommended default settings for the current hardware.

    Keys match the attribute names used in MainWindow / TranscriptionConfig.
    """
    profile = get_profile()
    vram = get_vram_gb()

    if profile == "apple_silicon":
        return {
            "whisper_device": "cpu",          # faster-whisper uses cpu; MLX Whisper handled separately
            "whisper_variant": "large-v3-turbo",
            "nllb_device": "cpu",
            "hf_summary_device": "cpu",
            "recommended_translation_model": "MLX Apple" if is_mlx_available() else "NLLB (lokalny)",
            "mlx_model_id": "mlx-community/gemma-3-12b-it-4bit",
            "profile_label": f"Apple Silicon ({get_gpu_name()}, {vram:.0f} GB unified memory)",
        }
    elif profile == "nvidia_gpu":
        # For large VRAM choose bigger/faster variant
        whisper_variant = "large-v3" if vram >= 16 else "large-v3-turbo"
        return {
            "whisper_device": "cuda",
            "whisper_variant": whisper_variant,
            "nllb_device": "cuda",
            "hf_summary_device": "cuda",
            "recommended_translation_model": "llama.cpp CUDA (lokalny)" if is_llama_cpp_available() else "NLLB (lokalny)",
            "llama_cpp_n_gpu_layers": -1,      # offload all layers to GPU
            "profile_label": f"NVIDIA GPU ({get_gpu_name()}, {vram:.1f} GB VRAM)",
        }
    elif profile == "amd_gpu":
        # AMD GPU via ROCm/HIP.  PyTorch built with ROCm maps device "cuda" to
        # the AMD GPU transparently, so HF summarization (pure PyTorch)
        # can use "cuda" here.
        #
        # faster-whisper and NLLB run on CTranslate2.  The PyPI wheels are
        # NVIDIA CUDA / CPU only; the ROCm wheels from the CTranslate2 GitHub
        # releases (rocm-python-wheels-*.zip) also report AMD GPUs as "cuda".
        # Use the GPU only when the installed CTranslate2 actually sees one.
        ct2_gpu = _ctranslate2_has_gpu()
        ct2_device = "cuda" if ct2_gpu else "cpu"
        ct2_label = "GPU" if ct2_gpu else "CPU (CTranslate2 bez ROCm)"
        whisper_variant = "large-v3" if vram >= 16 else "large-v3-turbo"
        return {
            "whisper_device": ct2_device,
            "whisper_variant": whisper_variant,
            "nllb_device": ct2_device,
            "hf_summary_device": "cuda",        # PyTorch ROCm build handles this fine
            "recommended_translation_model": "llama.cpp ROCm (lokalny)" if is_llama_cpp_available() else "NLLB (lokalny)",
            "llama_cpp_n_gpu_layers": -1,      # offload all layers to GPU
            "profile_label": f"AMD GPU ROCm ({get_gpu_name()}, {vram:.1f} GB pamięci) — Whisper/NLLB: {ct2_label}, streszczenia: GPU",
        }
    else:
        return {
            "whisper_device": "cpu",
            "whisper_variant": "medium",
            "nllb_device": "cpu",
            "hf_summary_device": "cpu",
            "recommended_translation_model": "NLLB (lokalny)",
            "profile_label": "CPU",
        }

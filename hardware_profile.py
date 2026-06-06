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

    # --- NVIDIA CUDA ---
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
                gpu_name = "NVIDIA GPU"
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
    else:
        return {
            "whisper_device": "cpu",
            "whisper_variant": "medium",
            "nllb_device": "cpu",
            "hf_summary_device": "cpu",
            "recommended_translation_model": "NLLB (lokalny)",
            "profile_label": "CPU",
        }

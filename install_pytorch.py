"""Automatyczny instalator PyTorch dla LinguaBee.

Wykrywa platformę i GPU, a następnie instaluje właściwy build PyTorch:
  - NVIDIA GPU              -> wheele CUDA z download.pytorch.org (cu128)
  - AMD GPU (RDNA2/3/3.5)   -> wheele ROCm z nightly multi-arch index
                               (projekt TheRock, https://github.com/ROCm/TheRock)
  - Apple Silicon (arm64)   -> standardowe wheele PyPI (Metal/MPS wbudowane)
  - Brak GPU / nierozpoznane -> lekkie wheele CPU-only z download.pytorch.org

WAŻNE: uruchom ten skrypt PRZED `pip install -r requirements.txt`, żeby
torch był już zainstalowany z właściwym backendem zanim pip zacznie
rozwiązywać zależności innych pakietów (transformers, whisperx,
pyannote.audio), które też wymagają torch.

Użycie:
    python install_pytorch.py                  # wykryj i zainstaluj
    python install_pytorch.py --dry-run         # tylko pokaż wykryty profil/komendę
    python install_pytorch.py --force cuda      # wymuś backend: cuda|rocm|cpu|mps
    python install_pytorch.py --gfx gfx1151     # wymuś konkretny target ROCm
    python install_pytorch.py --list-targets    # pokaż mapowanie GPU -> gfx target
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys

CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
ROCM_INDEX_URL = "https://rocm.nightlies.amd.com/whl-multi-arch/"

# Podłańcuch nazwy GPU/CPU (lowercase) -> docelowy target gfx (AMD ROCm).
# Kolejność ma znaczenie: bardziej szczegółowe dopasowania wyżej.
AMD_GFX_MAP: list[tuple[str, str]] = [
    # Strix Halo / Ryzen AI Max (np. ASUS ProArt, Ryzen AI Max+ 395)
    ("ryzen ai max+ pro 395", "gfx1151"),
    ("ryzen ai max+ 395", "gfx1151"),
    ("ryzen ai max+ pro 390", "gfx1151"),
    ("ryzen ai max+ 390", "gfx1151"),
    ("ryzen ai max", "gfx1151"),
    ("radeon 8060s", "gfx1151"),
    ("radeon 8050s", "gfx1151"),
    # Krackan Point
    ("ryzen ai 9 hx 375", "gfx1150"),
    ("ryzen ai 9 hx 370", "gfx1150"),
    ("ryzen ai 7 350", "gfx1152"),
    ("radeon 820m", "gfx1153"),
    # Desktop RDNA4 / RDNA3.5 iGPU
    ("radeon rx 9070", "gfx1201"),
    ("ai pro r9700", "gfx1201"),
    ("r9600d", "gfx1201"),
    ("radeon rx 9060", "gfx1200"),
    # Desktop RDNA3
    ("radeon rx 7900", "gfx1100"),
    ("radeon pro w7900", "gfx1100"),
    ("radeon pro w7800", "gfx1100"),
    ("radeon rx 7800", "gfx1101"),
    ("radeon rx 7700", "gfx1101"),
    ("radeon pro v710", "gfx1101"),
    ("radeon pro w7700", "gfx1101"),
    ("radeon rx 7600", "gfx1102"),
    # Phoenix / Hawk Point iGPU (RDNA3)
    ("7840u", "gfx1103"),
    ("7940hs", "gfx1103"),
    ("ryzen 9 270", "gfx1103"),
    ("radeon 780m", "gfx1103"),
    ("radeon 760m", "gfx1103"),
    # Desktop RDNA2
    ("radeon rx 6900", "gfx1030"),
    ("radeon rx 6800", "gfx1030"),
    ("radeon pro w6800", "gfx1030"),
    ("radeon pro v620", "gfx1030"),
    ("radeon rx 6750", "gfx1031"),
    ("radeon rx 6700", "gfx1031"),
    ("radeon rx 6600", "gfx1032"),
    ("radeon pro w6600", "gfx1032"),
    ("radeon rx 6500", "gfx1034"),
    ("radeon 680m", "gfx1035"),
    ("radeon 610m", "gfx1036"),
]


def _run(cmd: str, timeout: int = 15) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL, timeout=timeout
        ).decode(errors="ignore")
    except Exception:
        return ""


def _get_windows_device_names() -> list[str]:
    names: list[str] = []
    gpu_out = _run(
        'powershell -NoProfile -Command '
        '"Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"'
    )
    names += [line.strip() for line in gpu_out.splitlines() if line.strip()]
    cpu_out = _run(
        'powershell -NoProfile -Command '
        '"Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"'
    )
    names += [line.strip() for line in cpu_out.splitlines() if line.strip()]
    return names


def _get_linux_device_names() -> list[str]:
    names: list[str] = []
    lspci_out = _run("lspci -mm 2>/dev/null | grep -iE 'vga|3d|display'")
    names += [line.strip() for line in lspci_out.splitlines() if line.strip()]
    cpuinfo_out = _run("grep -m1 'model name' /proc/cpuinfo")
    if cpuinfo_out.strip():
        names.append(cpuinfo_out.strip())
    return names


def detect_gpu_vendor_and_target() -> tuple[str, str | None]:
    """Zwraca (vendor, gfx_target) gdzie vendor in {'nvidia','amd','apple','none'}."""
    system = platform.system()

    if system == "Darwin" and platform.machine() == "arm64":
        return "apple", None

    if system == "Windows":
        names = _get_windows_device_names()
    elif system == "Linux":
        names = _get_linux_device_names()
    else:
        names = []

    joined = " | ".join(names).lower()

    if "nvidia" in joined:
        return "nvidia", None

    if "amd" in joined or "radeon" in joined or "ryzen" in joined:
        # Kilka kart AMD (np. iGPU + eGPU) -> wszystkie targety, rozdzielone przecinkiem
        targets: list[str] = []
        for needle, gfx in AMD_GFX_MAP:
            if needle in joined and gfx not in targets:
                targets.append(gfx)
        if targets:
            return "amd", ",".join(targets)
        # Rozpoznano AMD, ale nieznany model -> bezpieczny fallback "wszystkie GPU"
        return "amd", None

    return "none", None


def build_install_command(
    vendor: str, gfx: str | None
) -> list[str]:
    py = sys.executable

    if vendor == "nvidia":
        return [
            py, "-m", "pip", "install",
            "--extra-index-url", CUDA_INDEX_URL,
            "torch", "torchvision", "torchaudio",
        ]

    if vendor == "amd":
        extra = ",".join(f"device-{g.strip()}" for g in gfx.split(",")) if gfx else "device-all"
        return [
            py, "-m", "pip", "install",
            "--index-url", ROCM_INDEX_URL,
            f"torch[{extra}]", f"torchvision[{extra}]", "torchaudio",
        ]

    if vendor == "apple":
        return [py, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]

    # "none" / CPU fallback
    return [
        py, "-m", "pip", "install",
        "--index-url", CPU_INDEX_URL,
        "torch", "torchvision", "torchaudio",
    ]


def _profile_label(vendor: str, gfx: str | None) -> str:
    if vendor == "nvidia":
        return "NVIDIA GPU -> PyTorch CUDA (cu128)"
    if vendor == "amd":
        target = gfx or "device-all (nierozpoznany model, bezpieczny fallback)"
        return f"AMD GPU -> PyTorch ROCm ({target})"
    if vendor == "apple":
        return "Apple Silicon -> PyTorch standardowy (Metal/MPS wbudowane)"
    return "Brak GPU / nierozpoznane -> PyTorch CPU-only"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", choices=["cuda", "rocm", "cpu", "mps"], default=None,
        help="Wymuś backend zamiast automatycznego wykrywania.",
    )
    parser.add_argument(
        "--gfx", default=None,
        help="Wymuś konkretny target ROCm (np. gfx1151), używane tylko z --force rocm "
             "lub gdy automatyczna detekcja AMD nie rozpozna modelu.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Tylko pokaż wykryty profil i komendę instalacji, nic nie instaluj.",
    )
    parser.add_argument(
        "--list-targets", action="store_true",
        help="Wypisz znane mapowania nazw GPU/CPU na targety ROCm i zakończ.",
    )
    args = parser.parse_args()

    if args.list_targets:
        for needle, gfx in AMD_GFX_MAP:
            print(f"  {needle!r:40s} -> {gfx}")
        return 0

    if args.force:
        vendor_map = {"cuda": "nvidia", "rocm": "amd", "cpu": "none", "mps": "apple"}
        vendor = vendor_map[args.force]
        gfx = args.gfx
    else:
        vendor, gfx = detect_gpu_vendor_and_target()
        if args.gfx:
            gfx = args.gfx

    print(f"== LinguaBee: instalator PyTorch ==")
    print(f"System: {platform.system()} ({platform.machine()})")
    print(f"Wykryty profil: {_profile_label(vendor, gfx)}")

    cmd = build_install_command(vendor, gfx)
    print("Komenda instalacji:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("(--dry-run) Nie instaluję.")
        return 0

    if vendor == "amd" and gfx is None:
        print(
            "Uwaga: nie rozpoznano dokładnego modelu GPU AMD - instaluję wariant "
            "'device-all' (obsługuje wszystkie GPU, ale większe pobieranie). "
            "Możesz podać dokładny target ręcznie przez --gfx (zobacz --list-targets)."
        )

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

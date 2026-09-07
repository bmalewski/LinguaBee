import sys
import os
import multiprocessing
from pathlib import Path

# PyInstaller: musi być wywołane zanim cokolwiek uruchomi proces potomny.
multiprocessing.freeze_support()

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config  # tworzy katalogi zapisu i (w spakowanej aplikacji) seeduje szablony promptów


def _bootstrap_runtime_env() -> None:
    """Tylko w spakowanej aplikacji: wbudowane ffmpeg/ffprobe na PATH i zapisywalny cache numba.

    Aplikacja uruchomiona dwuklikiem dziedziczy PATH launchd bez /opt/homebrew/bin,
    a katalog aplikacji może być tylko do odczytu.
    """
    if not config.is_frozen():
        return
    extra = [p for p in (config.bin_dir, "/opt/homebrew/bin", "/usr/local/bin") if os.path.isdir(p)]
    os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])
    if sys.platform == "darwin":
        cache_root = os.path.expanduser(f"~/Library/Caches/{config.APP_NAME}")
    else:
        cache_root = os.path.join(config.data_root, "cache")
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(cache_root, "numba"))


_bootstrap_runtime_env()

import warnings

# Ignoruj konkretne ostrzeżenie UserWarning z ctranslate2 dotyczące pkg_resources
# Ustawiamy filtr zanim zaimportujemy moduły, które mogą załadować ctranslate2
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated as an API.*")

# Compatibility shim: some versions of libraries expect `torchaudio.AudioMetaData` to
# be available at top-level. Recent torchaudio may expose it under backend modules.
# If it's missing, try to alias it from common/sox_io_backend so code in site-packages
# (pyannote, speechbrain, etc.) doesn't fail with "module 'torchaudio' has no attribute 'AudioMetaData'".
try:
    import torchaudio
    if not hasattr(torchaudio, 'AudioMetaData'):
        # try a few likely locations
        try:
            from torchaudio.backend import common as _ta_common
            if hasattr(_ta_common, 'AudioMetaData'):
                torchaudio.AudioMetaData = _ta_common.AudioMetaData
        except Exception:
            try:
                from torchaudio.backend import sox_io_backend as _sox
                if hasattr(_sox, 'AudioMetaData'):
                    torchaudio.AudioMetaData = _sox.AudioMetaData
            except Exception:
                # give up silently; downstream imports will raise their own errors
                pass
except Exception:
    # torchaudio not installed or import failed; nothing to do here
    pass


def _run_selftest() -> int:
    """Sprawdza, czy wszystkie leniwie importowane biblioteki i binarki są dostępne.

    Używane głównie do weryfikacji spakowanej aplikacji:
        LinguaBee.app/Contents/MacOS/LinguaBee --selftest
    """
    import importlib
    import shutil

    modules = [
        "PySide6.QtWidgets", "torch", "torchaudio", "torchcodec", "faster_whisper", "ctranslate2",
        "whisperx", "librosa", "pyannote.audio", "transformers", "sentencepiece", "sacremoses",
        "av", "soundfile", "pydub", "noisereduce", "yt_dlp", "docx", "httpx", "huggingface_hub",
        "onnxruntime", "numba", "tools.process_audio_runner",
    ]
    # torchcodec wymaga bibliotek współdzielonych ffmpeg (libav*.dylib). Diaryzacja go nie
    # potrzebuje (audio_processing dekoduje audio przez PyAV), więc brak nie jest błędem.
    optional = {"torchcodec"}
    failed = []
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"OK    {name}")
        except Exception as e:
            short = str(e).splitlines()[0][:160]
            if name in optional:
                print(f"WARN  {name} (opcjonalne): {short}")
            else:
                failed.append(name)
                print(f"FAIL  {name}: {short}")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        print(f"{'OK   ' if path else 'FAIL '} {tool} -> {path}")
        if not path:
            failed.append(tool)
    print(
        f"frozen: {config.is_frozen()} | downloads: {config.downloads_dir} | models: {config.models_dir} | "
        f"prompts: {config.prompts_dir} | settings: {config.settings_file}"
    )
    return 1 if failed else 0


# --- Tryby CLI: muszą być obsłużone PRZED importem PySide6 ---
# W spakowanej aplikacji sys.executable to binarka aplikacji, więc worker.py uruchamia
# izolowany runner audio jako `LinguaBee --audio-runner <plik>`.
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--audio-runner":
    from tools.process_audio_runner import main as _audio_runner_main
    sys.exit(_audio_runner_main(sys.argv[2:]))

if __name__ == "__main__" and "--selftest" in sys.argv[1:]:
    sys.exit(_run_selftest())

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from gui.main_window import MainWindow


def _load_stylesheet(app: QApplication):
    """Load and apply the bundled QSS stylesheet if present.

    This ensures the app uses the project's intended colors, fonts and widget styles.
    """
    try:
        qss_path = config.resource_path("gui", "stylesheet.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
    except Exception:
        # Non-fatal: continue without stylesheet if anything goes wrong
        pass


def _apply_readable_font(app: QApplication):
    """Apply a consistent, readable UI font across platforms.

    Segoe UI baseline profile for quick typography comparison.
    """
    try:
        app.setFont(QFont("Segoe UI", 13))
    except Exception:
        pass


def _load_custom_fonts(app: QApplication):
    """Load custom fonts from project font directory and apply SF Pro family.

    Supported files: .ttf, .otf, .ttc
    """
    try:
        font_dir = Path(config.resource_path("font"))
        fallback_sfpro_dir = font_dir / "SFPro"

        if not font_dir.exists() or not font_dir.is_dir():
            return

        font_files = []
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            font_files.extend(sorted(font_dir.rglob(ext)))

        if not font_files and fallback_sfpro_dir.exists() and fallback_sfpro_dir.is_dir():
            for ext in ("*.ttf", "*.otf", "*.ttc"):
                font_files.extend(sorted(fallback_sfpro_dir.rglob(ext)))

        if not font_files:
            return

        loaded_families = []
        for font_path in font_files:
            try:
                font_id = QFontDatabase.addApplicationFont(str(font_path))
                if font_id == -1:
                    continue
                families = QFontDatabase.applicationFontFamilies(font_id)
                loaded_families.extend([family for family in families if family])
            except Exception:
                continue

        if loaded_families:
            unique_families = []
            for family in loaded_families:
                if family not in unique_families:
                    unique_families.append(family)

            preferred_family = None
            preferred_order = ["SF Pro Rounded", "SF Pro", "SFPro", "SF Pro Text", "SF Pro Display"]
            for preferred_name in preferred_order:
                for family in unique_families:
                    if preferred_name.lower() in family.lower():
                        preferred_family = family
                        break
                if preferred_family:
                    break

            primary_family = preferred_family or unique_families[0]
            current_font = app.font()
            size = current_font.pointSize() if current_font.pointSize() > 0 else 11
            app.setFont(QFont(primary_family, size, 500))
    except Exception:
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    _apply_readable_font(app)
    # Wczytaj opcjonalne czcionki projektu (np. SF Pro), jeśli katalog font/ istnieje.
    _load_custom_fonts(app)
    # Zastosuj arkusz stylów projektu, aby kolory/czcionki pasowały do projektu interfejsu użytkownika
    _load_stylesheet(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

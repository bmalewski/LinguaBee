import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import warnings
from PySide6.QtWidgets import QMessageBox

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

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from gui.main_window import MainWindow


def _load_stylesheet(app: QApplication):
    """Load and apply the bundled QSS stylesheet if present.

    This ensures the app uses the project's intended colors, fonts and widget styles.
    """
    try:
        base = os.path.dirname(__file__)
        qss_path = os.path.join(base, "gui", "stylesheet.qss")
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
        base = Path(__file__).resolve().parent
        font_dir = base / "font"
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
    # Zastosuj arkusz stylów projektu, aby kolory/czcionki pasowały do projektu interfejsu użytkownika
    _load_stylesheet(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

import sys
import os

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Zastosuj arkusz stylów projektu, aby kolory/czcionki pasowały do projektu interfejsu użytkownika
    _load_stylesheet(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

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

# --- Start bez konsoli (pythonw.exe: LinguaBee.bat, skrót na pulpicie) ---
# pythonw nie ma konsoli, więc sys.stdout i sys.stderr to None. Paski postępu pobierania
# modeli (tqdm) i biblioteki piszące na stderr wywracałyby się wtedy, a błędy znikałyby
# bez śladu. Dlatego wyjście trafia do logs/linguabee.log w katalogu danych, a błąd przy
# starcie pokazuje okno z informacją, gdzie szukać szczegółów.
LOG_FILE = os.path.join(config.data_root, "logs", "linguabee.log")
_LOG_MAX_BYTES = 5 * 1024 * 1024
_NO_CONSOLE = sys.stdout is None or sys.stderr is None
_GUI_MODE = (
    __name__ == "__main__"
    and not (len(sys.argv) > 1 and sys.argv[1] == "--audio-runner")
    and "--selftest" not in sys.argv[1:]
)
_startup_done = False


def _redirect_output_to_log() -> None:
    """Pod pythonw kieruje stdout/stderr do pliku logu (poprzedni log > 5 MB -> .log.1)."""
    if not _NO_CONSOLE:
        return
    stream = None
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > _LOG_MAX_BYTES:
            os.replace(LOG_FILE, LOG_FILE + ".1")
        stream = open(LOG_FILE, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    # Paski postępu pobierania modeli zaśmiecałyby plik logu.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import faulthandler  # zrzut stosu przy awarii bibliotek natywnych (CUDA, ctranslate2)
        faulthandler.enable(file=stream)
    except Exception:
        pass
    import time
    print(f"\n=== LinguaBee {time.strftime('%Y-%m-%d %H:%M:%S')} | {sys.executable} | {sys.argv[1:]} ===")


def _show_error_box(title: str, text: str) -> None:
    """Okno błędu bez Qt: działa nawet wtedy, gdy PySide6 się nie załadował."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10 | 0x10000)  # MB_ICONERROR | MB_SETFOREGROUND
    except Exception:
        pass


def _excepthook(exc_type, exc, tb) -> None:
    """Nieobsłużony wyjątek: traceback do logu, a przy starcie bez konsoli także okno błędu."""
    import traceback
    traceback.print_exception(exc_type, exc, tb)
    if _NO_CONSOLE and not _startup_done and not issubclass(exc_type, KeyboardInterrupt):
        _show_error_box(
            "LinguaBee – błąd uruchomienia",
            f"LinguaBee nie może się uruchomić.\n\n{exc_type.__name__}: {exc}\n\n"
            f"Szczegóły zapisano w pliku:\n{LOG_FILE}\n\n"
            "Pełne komunikaty zobaczysz po uruchomieniu pliku „LinguaBee - konsola.bat”.",
        )


_redirect_output_to_log()
if _GUI_MODE:
    sys.excepthook = _excepthook

# AMD ROCm: przy kilku widocznych kartach PyTorch się wywraca (0xC0000005),
# więc procesowi pokazujemy tylko jedną kartę: wybraną w ustawieniach
# ('amd_gpu', domyślnie 0) albo podaną ręcznie w HIP_VISIBLE_DEVICES.
# Musi być ustawione przed pierwszym importem torch/ctranslate2.
if "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = str(config.load_settings().get("amd_gpu", 0))


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

import threading
import warnings

# Ignoruj konkretne ostrzeżenie UserWarning z ctranslate2 dotyczące pkg_resources
# Ustawiamy filtr zanim zaimportujemy moduły, które mogą załadować ctranslate2
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated as an API.*")

# Szybki start: ciężkie biblioteki (PyTorch, transformers, Whisper, yt-dlp) nie są tu
# importowane. Najpierw pojawia się ekran startowy, potem okno (PyTorch jest potrzebny tylko
# do wykrycia karty graficznej), a biblioteki przetwarzania ładują się w tle, gdy okno już
# działa (worker.load_backends). Poprawka torchaudio dla pyannote: torchaudio_compat.py.


def _run_selftest() -> int:
    """Sprawdza, czy wszystkie leniwie importowane biblioteki i binarki są dostępne.

    Używane głównie do weryfikacji spakowanej aplikacji:
        LinguaBee.app/Contents/MacOS/LinguaBee --selftest
    """
    import importlib
    import shutil
    from torchaudio_compat import ensure_audio_metadata

    ensure_audio_metadata()  # jak w programie: przed importem pyannote/whisperx
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
    # Moduły LinguaBee ładowane dopiero po pokazaniu okna (w tle albo po kliknięciu Start):
    # ich błąd wyszedłby inaczej dopiero przy pierwszym zadaniu. Instalator uruchamia selftest
    # na koniec instalacji, więc przy okazji powstają ich pliki .pyc i pierwszy start jest szybszy.
    try:
        importlib.import_module("worker").load_backends()
        importlib.import_module("gui.main_window")
        print("OK    moduły LinguaBee (przetwarzanie i okno)")
    except Exception as e:
        failed.append("moduły LinguaBee")
        print(f"FAIL  moduły LinguaBee: {str(e).splitlines()[0][:160]}")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        print(f"{'OK   ' if path else 'FAIL '} {tool} -> {path}")
        if not path:
            failed.append(tool)
    try:
        import torch
        if torch.cuda.is_available():
            hip = getattr(torch.version, "hip", None)
            backend = f"ROCm {hip}" if hip else f"CUDA {torch.version.cuda}"
            print(f"OK    GPU: {backend} -> {torch.cuda.get_device_name(0)}")
        else:
            print("INFO  GPU niedostępne (obliczenia na CPU)")
    except Exception as e:
        print(f"WARN  GPU: {str(e).splitlines()[0][:160]}")
    mode = f"LinguaBee {config.APP_VERSION} | tryb: {config.run_mode()}"
    if config.is_installed():
        mode += f" | instalacja: {config.install_root} (wariant: {config.install_info.get('variant', '?')})"
    print(mode)
    print(
        f"frozen: {config.is_frozen()} | downloads: {config.downloads_dir} | models: {config.models_dir} | "
        f"prompts: {config.prompts_dir} | settings: {config.settings_file}"
    )
    print(f"python: {sys.executable} | log (start bez konsoli): {LOG_FILE}")
    return 1 if failed else 0


# --- Tryby CLI: muszą być obsłużone PRZED importem PySide6 ---
# W spakowanej aplikacji sys.executable to binarka aplikacji, więc worker.py uruchamia
# izolowany runner audio jako `LinguaBee --audio-runner <plik>`.
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--audio-runner":
    from torchaudio_compat import ensure_audio_metadata
    ensure_audio_metadata()
    from tools.process_audio_runner import main as _audio_runner_main
    sys.exit(_audio_runner_main(sys.argv[2:]))

if __name__ == "__main__" and "--selftest" in sys.argv[1:]:
    sys.exit(_run_selftest())

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen, QPixmap

_SPLASH_WIDTH, _SPLASH_HEIGHT = 420, 320


class _StartupSplash(QSplashScreen):
    """Ekran startowy: logo, wersja i bieżący etap uruchamiania."""

    def drawContents(self, painter):
        painter.setPen(QColor("#66737C"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(QRectF(0, _SPLASH_HEIGHT - 56, _SPLASH_WIDTH, 32), Qt.AlignCenter, self.message())


def _show_splash(app: QApplication):
    """Pokazuje ekran startowy po ułamku sekundy od kliknięcia (jak okno ładowania ComfyUI).

    Okno programu potrzebuje PyTorch (wykrycie karty graficznej), który ładuje się od 2 s
    do kilkunastu sekund po starcie komputera. Bez ekranu startowego przez ten czas nic by
    się nie działo. Ekran jest tylko dodatkiem: błąd przy jego tworzeniu nie blokuje startu.
    """
    try:
        screen = app.primaryScreen()
        dpr = screen.devicePixelRatio() if screen is not None else 1.0
        canvas = QPixmap(round(_SPLASH_WIDTH * dpr), round(_SPLASH_HEIGHT * dpr))
        canvas.setDevicePixelRatio(dpr)
        canvas.fill(QColor("#FFFFFF"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setPen(QPen(QColor("#CFD8DC"), 1))
        painter.drawRect(QRectF(0.5, 0.5, _SPLASH_WIDTH - 1, _SPLASH_HEIGHT - 1))
        logo = QPixmap(os.path.join(config.icons_dir, "LinguaBee_logo_transparent.png"))
        if not logo.isNull():
            logo = logo.scaledToHeight(round(180 * dpr), Qt.SmoothTransformation)
            logo.setDevicePixelRatio(dpr)
            painter.drawPixmap(QPointF((_SPLASH_WIDTH - logo.width() / dpr) / 2, 26), logo)
        painter.setPen(QColor("#546E7A"))  # nazwa jest już w logo
        painter.setFont(QFont("Segoe UI", 11))
        painter.drawText(QRectF(0, 216, _SPLASH_WIDTH, 30), Qt.AlignCenter, f"wersja {config.APP_VERSION}")
        painter.end()

        splash = _StartupSplash(canvas)
        splash.setWindowTitle("LinguaBee – uruchamianie")  # niewidoczny; dla czytników ekranu
        splash.show()
        _splash_message(app, splash, "Uruchamianie…")
        return splash
    except Exception:
        return None


def _splash_message(app: QApplication, splash, text: str) -> None:
    if splash is not None:
        splash.showMessage(text)
        app.processEvents()


def _run_with_live_splash(app: QApplication, func) -> None:
    """Wykonuje func w osobnym wątku, a wątek główny w tym czasie obsługuje ekran startowy.

    Import PyTorch potrafi trwać kilkanaście sekund; gdyby blokował wątek okien, Windows
    uznałby ekran startowy za „nieodpowiadający”. Błąd jest zgłaszany tutaj, w wątku
    głównym, więc obsługuje go zwykła ścieżka błędów startu (_excepthook).
    """
    errors = []

    def target():
        try:
            func()
        except BaseException as exc:  # przekazywany dalej w wątku głównym
            errors.append(exc)

    thread = threading.Thread(target=target, name="startup-import", daemon=True)
    thread.start()
    while thread.is_alive():
        app.processEvents()
        thread.join(0.03)
    if errors:
        raise errors[0]


def _prepare_main_window_module() -> None:
    """Import gui.main_window (PyTorch, wykrycie karty) i odczyt listy kart, bez tworzenia okna."""
    import importlib
    importlib.import_module("gui.main_window").prefetch_hardware_info()


def _start_background_preload() -> None:
    """Ładuje w tle biblioteki przetwarzania (worker.load_backends), gdy okno już działa.

    Zanim użytkownik wybierze plik i kliknie Start, biblioteki są zwykle gotowe. Start kliknięty
    wcześniej poczeka na koniec ładowania (z komunikatem w dzienniku okna).
    """
    def target():
        import time
        started = time.perf_counter()
        try:
            import worker
            worker.load_backends()
            print(f"Biblioteki przetwarzania załadowane w tle w {time.perf_counter() - started:.1f} s.")
        except Exception:
            import traceback
            print("Nie udało się załadować bibliotek przetwarzania w tle (Start pokaże błąd):")
            traceback.print_exc()

    threading.Thread(target=target, name="preload-backends", daemon=True).start()


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
    splash = _show_splash(app)
    try:
        _apply_readable_font(app)
        # Wczytaj opcjonalne czcionki projektu (np. SF Pro), jeśli katalog font/ istnieje.
        _load_custom_fonts(app)
        # Zastosuj arkusz stylów projektu, aby kolory/czcionki pasowały do projektu interfejsu użytkownika
        _load_stylesheet(app)
        _splash_message(app, splash, "Ładowanie PyTorch i wykrywanie karty graficznej…")
        _run_with_live_splash(app, _prepare_main_window_module)
        _splash_message(app, splash, "Przygotowanie okna…")
        from gui.main_window import MainWindow
        window = MainWindow()
    except BaseException:
        if splash is not None:
            splash.close()  # żeby nie zasłaniał okna błędu
        raise
    window.show()
    if splash is not None:
        splash.finish(window)
    # Od tej chwili błędy trafiają tylko do logu; okno błędu dotyczy wyłącznie startu.
    _startup_done = True
    # Biblioteki przetwarzania: w tle, gdy okno już się narysowało.
    QTimer.singleShot(500, _start_background_preload)
    sys.exit(app.exec())

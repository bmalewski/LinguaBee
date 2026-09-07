# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: LinguaBee jako aplikacja macOS (.app), one-dir, arm64.

Budowanie: packaging/build_macos.sh (instaluje PyInstaller, pobiera statyczne
ffmpeg/ffprobe, generuje ikonę .icns, uruchamia PyInstaller i podpisuje ad-hoc).

Zasoby tylko do odczytu trafiają do sys._MEIPASS (Contents/Frameworks) i są czytane
przez config.resource_path(). Dane użytkownika (wyniki, modele, ustawienia, szablony)
trafiają do katalogów systemowych — patrz config.py.
"""
import os

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ASSETS = os.path.join(ROOT, "build_assets")
APP_VERSION = "1.0.2"

# --- Zasoby tylko do odczytu (config.resource_path) ---
datas = [
    (os.path.join(ROOT, "icons", "LinguaBee_1.0_512.png"), "icons"),
    (os.path.join(ROOT, "gui", "stylesheet.qss"), "gui"),
    # Seedy szablonów promptów — kopiowane do Application Support przy pierwszym uruchomieniu.
    (os.path.join(ROOT, "prompts"), "prompts"),
]

# --- Wbudowane binarki (statyczne ffmpeg/ffprobe arm64) ---
binaries = [
    (os.path.join(ASSETS, "bin", "ffmpeg"), "bin"),
    (os.path.join(ASSETS, "bin", "ffprobe"), "bin"),
]

# --- Importy niewidoczne dla analizy statycznej ---
hiddenimports = [
    # moduły projektu importowane leniwie (wewnątrz funkcji / try-except)
    "tools", "tools.process_audio_runner", "whisper_aligner", "translation_manager",
    "audio_processing", "helsinki_translator", "mlx_translator", "llama_cpp_translator",
    "api_client", "bart_summarizer", "hardware_profile", "nllb_translator",
    "ollama_translator", "ollama_summarizer", "ollama_refiner", "summarization_manager",
    "whisper_paragrafizer", "correction_service", "correction_adapters",
    # biblioteki importowane leniwie lub przez dynamiczny dispatch
    "whisperx", "librosa", "sentencepiece", "sacremoses",
    "asteroid_filterbanks", "lightning", "pytorch_lightning", "torchmetrics", "omegaconf",
    "einops", "julius", "torch_audiomentations", "torchcodec", "safetensors", "tokenizers",
    "hf_xet", "certifi", "pytorch_metric_learning", "soxr", "pooch", "lazy_loader",
]

# Pakiety z dynamicznymi importami lub zasobami niebędącymi kodem Pythona.
# collect_all = podmoduły + dane + biblioteki natywne + metadane.
# Celowo BEZ "torch": hook z pyinstaller-hooks-contrib wystarcza, a collect_all
# wciągnąłby setki MB źródeł (inductor, testy).
for _pkg in [
    "transformers", "pyannote.audio", "pyannote.core", "pyannote.pipeline", "pyannote.database",
    "pyannote.metrics", "whisperx", "faster_whisper", "librosa", "lightning", "pytorch_lightning",
    "torchmetrics", "torchaudio", "torchcodec", "ctranslate2", "yt_dlp", "docx",
    "soundfile", "av", "onnxruntime", "numba", "llvmlite", "huggingface_hub", "sentencepiece",
    "sacremoses", "nltk", "noisereduce", "pydub",
]:
    try:
        _d, _b, _h = collect_all(_pkg)
    except Exception as _e:  # pakiet niezainstalowany — pomiń, ale pokaż w logu
        print(f"[spec] collect_all({_pkg!r}) pominięte: {_e}")
        continue
    datas += _d
    binaries += _b
    hiddenimports += _h

binaries += collect_dynamic_libs("torch")

# --- Wykluczenia: projekt używa tylko QtCore/QtGui/QtWidgets ---
excludes = [
    *[
        f"PySide6.{m}"
        for m in (
            "QtWebEngineCore", "QtWebEngineWidgets", "QtWebEngineQuick", "QtWebChannel",
            "QtWebSockets", "QtWebView", "Qt3DCore", "Qt3DRender", "Qt3DInput", "Qt3DLogic",
            "Qt3DAnimation", "Qt3DExtras", "QtQml", "QtQuick", "QtQuick3D", "QtQuickControls2",
            "QtQuickWidgets", "QtQuickTest", "QtCharts", "QtDataVisualization", "QtGraphs",
            "QtGraphsWidgets", "QtMultimedia", "QtMultimediaWidgets", "QtSpatialAudio",
            "QtTextToSpeech", "QtPdf", "QtPdfWidgets", "QtLocation", "QtPositioning",
            "QtSensors", "QtSerialPort", "QtSerialBus", "QtBluetooth", "QtNfc", "QtDesigner",
            "QtHelp", "QtSql", "QtTest", "QtRemoteObjects", "QtScxml", "QtStateMachine",
            "QtHttpServer", "QtNetworkAuth", "QtUiTools", "QtOpenGL", "QtOpenGLWidgets",
            "QtPrintSupport", "QtSvgWidgets", "QtXml", "QtConcurrent",
        )
    ],
    "tkinter", "IPython", "jupyter", "notebook", "pytest", "sphinx",
    # matplotlib/pandas/sklearn/torchvision zostają: używa ich pyannote.metrics;
    # sympy zostaje: wymaga go torch.
]

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LinguaBee",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LinguaBee",
)

app = BUNDLE(
    coll,
    name="LinguaBee.app",
    icon=os.path.join(ASSETS, "LinguaBee.icns"),
    bundle_identifier="pl.malewski.linguabee",
    info_plist={
        "CFBundleName": "LinguaBee",
        "CFBundleDisplayName": "LinguaBee",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "LSMinimumSystemVersion": "14.0",
        "LSApplicationCategoryType": "public.app-category.productivity",
    },
)

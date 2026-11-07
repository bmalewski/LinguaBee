import os
import sys
from dataclasses import dataclass
from typing import Optional, List
import json

# -------------------------------------------------
# Paths
# -------------------------------------------------
# Dwa rodzaje ścieżek:
# - zasoby tylko do odczytu (ikony, stylesheet, seedy promptów, wbudowane binarki):
#   resource_path(...) — w spakowanej aplikacji (PyInstaller) wskazuje do sys._MEIPASS,
#   poza nią do katalogu z kodem;
# - dane użytkownika (wyniki, modele, ustawienia, szablony promptów, cache):
#   * ze źródeł (python main.py, LinguaBee.bat w repozytorium) — jak dotąd obok kodu;
#   * w instalacji z instalatora Windows (kod w <root>\app, obok plik install.json) —
#     wszystko w <root>: models\, output\, user\ (ustawienia, prompty, logi), cache\, bin\;
#   * w spakowanej aplikacji — katalogi standardowe systemu (macOS: ~/Documents/LinguaBee
#     i ~/Library/Application Support/LinguaBee).
import shutil

APP_NAME = "LinguaBee"
APP_VERSION = "1.1.0"  # jedyne źródło numeru wersji (instalator Windows, selftest)
INSTALL_MARKER = "install.json"
_SOURCE_ROOT = os.path.dirname(os.path.abspath(__file__))


def is_frozen() -> bool:
    """True, gdy aplikacja działa jako spakowany plik wykonywalny (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def resource_path(*parts) -> str:
    """Ścieżka do zasobu tylko do odczytu dołączonego do aplikacji."""
    root = getattr(sys, "_MEIPASS", None) if is_frozen() else None
    return os.path.join(root or _SOURCE_ROOT, *parts)


def _find_install_root() -> Optional[str]:
    """Katalog główny instalacji z instalatora Windows albo None (źródła, spakowana aplikacja).

    Instalator kładzie kod w <root>\\app, a obok niego install.json. Zmienna środowiskowa
    LINGUABEE_HOME wymusza katalog główny (np. do testów).
    """
    forced = os.environ.get("LINGUABEE_HOME")
    if forced:
        return os.path.abspath(forced)
    if is_frozen():
        return None
    parent = os.path.dirname(_SOURCE_ROOT)
    if os.path.isfile(os.path.join(parent, INSTALL_MARKER)):
        return parent
    return None


def _load_install_info(root: Optional[str]) -> dict:
    """Zawartość install.json (wariant, wersje, opcjonalne models_dir / output_dir)."""
    if not root:
        return {}
    try:
        # utf-8-sig: PowerShell 5.1 zapisuje UTF-8 z BOM
        with open(os.path.join(root, INSTALL_MARKER), "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


install_root = _find_install_root()
install_info = _load_install_info(install_root)


def is_installed() -> bool:
    """True, gdy program działa z instalacji (wszystkie dane w jednym folderze)."""
    return install_root is not None


def run_mode() -> str:
    if install_root:
        return "instalacja"
    if is_frozen():
        return "spakowana aplikacja"
    return "źródła"


def _install_path(key: str, default_name: str) -> str:
    """Ścieżka w folderze instalacji. install.json może ją przenieść gdzie indziej
    (np. "models_dir": "E:\\\\Modele") — odpowiednik extra_model_paths.yaml z ComfyUI."""
    custom = install_info.get(key)
    if isinstance(custom, str) and custom.strip():
        return os.path.abspath(os.path.expandvars(custom.strip()))
    return os.path.join(install_root, default_name)


def _user_data_root() -> str:
    """Katalog zapisu dla ustawień i szablonów promptów (oraz modeli poza instalacją)."""
    if install_root:
        return os.path.join(install_root, "user")
    if not is_frozen():
        return _SOURCE_ROOT
    if sys.platform == "darwin":
        return os.path.expanduser(f"~/Library/Application Support/{APP_NAME}")
    if sys.platform.startswith("win"):
        return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), APP_NAME)
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), APP_NAME)


def _user_output_root() -> str:
    """Katalog, do którego trafiają wyniki (TXT/DOCX/SRT, pobrane audio)."""
    if install_root:
        return _install_path("output_dir", "output")
    if not is_frozen():
        return os.path.join(_SOURCE_ROOT, "output")
    return os.path.join(os.path.expanduser("~"), "Documents", APP_NAME)


def _safe_makedirs(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


# Stałe publiczne — nazwy niezmienione, importowane w wielu modułach.
base_path = os.path.dirname(sys.executable) if is_frozen() else _SOURCE_ROOT
data_root = _user_data_root()
downloads_dir = _user_output_root()
models_dir = _install_path("models_dir", "models") if install_root else os.path.join(data_root, "models")
prompts_dir = os.path.join(data_root, "prompts")
settings_file = os.path.join(data_root, "user_settings.json")
icons_dir = resource_path("icons")
# ffmpeg/ffprobe (i uv, deno): w instalacji <root>\bin, w spakowanej aplikacji wbudowane
bin_dir = os.path.join(install_root, "bin") if install_root else resource_path("bin")
cache_dir = os.path.join(install_root, "cache") if install_root else os.path.join(data_root, "cache")


def prompts_subdir(*parts) -> str:
    """Zapisywalny katalog szablonów promptów (tworzony na żądanie)."""
    path = os.path.join(prompts_dir, *parts)
    _safe_makedirs(path)
    return path


def _seed_prompts() -> None:
    """Spakowana aplikacja i instalacja: kopiuje dołączone szablony promptów do katalogu
    użytkownika (user\\prompts), żeby aktualizacja kodu ich nie nadpisała. Nigdy nie
    nadpisuje istniejących plików."""
    if not (is_frozen() or install_root):
        return
    src_root = resource_path("prompts")
    if not os.path.isdir(src_root) or os.path.normcase(os.path.abspath(src_root)) == os.path.normcase(
            os.path.abspath(prompts_dir)):
        return
    for dirpath, _dirs, files in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        dst_dir = prompts_dir if rel == "." else os.path.join(prompts_dir, rel)
        _safe_makedirs(dst_dir)
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            dst = os.path.join(dst_dir, fn)
            if os.path.exists(dst):
                continue
            try:
                shutil.copyfile(os.path.join(dirpath, fn), dst)
            except OSError:
                pass


_dll_directories = []  # uchwyty os.add_dll_directory (muszą żyć do końca procesu)


def _configure_install_environment() -> None:
    """Instalacja: modele, cache i narzędzia wyłącznie w folderze instalacji.

    Musi zadziałać przed pierwszym importem huggingface_hub / torch / yt_dlp (main.py importuje
    config jako pierwszy). Procesy potomne (np. runner audio) dziedziczą te zmienne.
    Wartości ustawione wcześniej w środowisku mają pierwszeństwo.
    """
    if not install_root:
        return
    for key, value in {
        "HF_HOME": os.path.join(data_root, "huggingface"),  # token HuggingFace
        "HF_HUB_CACHE": models_dir,  # modele HF w tym samym folderze co Whisper i NLLB
        "TORCH_HOME": os.path.join(cache_dir, "torch"),
        "XDG_CACHE_HOME": cache_dir,  # m.in. cache yt-dlp
        "MPLCONFIGDIR": os.path.join(cache_dir, "matplotlib"),
        "NUMBA_CACHE_DIR": os.path.join(cache_dir, "numba"),
        "DENO_DIR": os.path.join(cache_dir, "deno"),
    }.items():
        os.environ.setdefault(key, value)
    if os.path.isdir(bin_dir):
        path = os.environ.get("PATH", "")
        if os.path.normcase(bin_dir) not in [os.path.normcase(p) for p in path.split(os.pathsep)]:
            os.environ["PATH"] = bin_dir + os.pathsep + path
        if hasattr(os, "add_dll_directory"):  # DLL-e FFmpeg dla torchcodec
            try:
                _dll_directories.append(os.add_dll_directory(bin_dir))
            except OSError:
                pass


# pyannote.audio domyślnie wysyła anonimowe statystyki użycia; wyłączamy (można nadpisać zmienną).
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

for _d in (downloads_dir, models_dir, prompts_dir):
    _safe_makedirs(_d)
_configure_install_environment()
_seed_prompts()


def load_settings() -> dict:
    """Load user settings from JSON file. Returns a dict (possibly empty)."""
    try:
        if os.path.exists(settings_file):
            with open(settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


def save_settings(settings: dict) -> bool:
    """Save user settings to JSON file. Overwrites existing file."""
    try:
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# -------------------------------------------------
# Data Classes for Configuration
# -------------------------------------------------
@dataclass
class TranscriptionConfig:
    """Holds all the configuration for a transcription and translation job."""
    url: str
    local_files: List[str]
    transcription_model: str
    whisper_variant: str
    whisper_device: str
    whisper_device_index: int
    translation_model: str
    nllb_variant: str
    nllb_device: str
    nllb_device_index: int
    hf_summary_model_name: str
    hf_summary_device: str
    hf_summary_device_index: int
    hf_summary_max_length: int
    hf_summary_min_length: int
    hf_summary_num_beams: int
    ollama_model_name: str
    translation_openrouter_model_name: str
    # CTranslate2 summarization model options
    ctranslate2_device_index: int
    ctranslate2_tokenizer_name: Optional[str]
    ctranslate2_max_input_tokens: int
    ctranslate2_max_decoding_length: int
    ctranslate2_beam_size: int
    summary_model: str
    ollama_summary_model_name: str
    src_lang_code: str
    translation_src_lang_code: str
    tgt_lang_code: str
    summary_lang_code: str
    formats_original: List[str]
    formats_translated: List[str]
    formats_summary: List[str]
    openai_key: Optional[str]
    gemini_key: Optional[str]
    openrouter_key: Optional[str]
    delete_audio: bool
    # WhisperX integration flags (optional)
    enable_whisperx: bool = False
    whisperx_diarization: bool = False
    # If you want to force a device for whisperx alignment/diarization (e.g. 'cuda' or 'cpu')
    whisperx_device: Optional[str] = None
    # Optional diarization model name (left None to use default pyannote pipeline if available)
    whisperx_diarization_model: Optional[str] = None
    # Paragraphing options
    enable_paragraphing: bool = False
    paragraph_silence_threshold: float = 1.0
    paragraph_max_chars: int = 500
    paragraph_break_on_speaker: bool = True
    paragraph_min_sentence_chars: int = 10
    # Diarization options
    enable_diarization: bool = False
    hf_token: Optional[str] = None
    num_speakers: int = 0
    # Denoising option
    enable_denoising: bool = False
    enable_normalization: bool = False
    force_mono: bool = False
    # Limit czasu (w sekundach) na izolowany proces przetwarzania audio
    audio_processing_timeout: int = 300
    # SRT line formatting options for transcription output
    srt_max_lines: int = 2
    srt_max_chars_per_line: int = 25
    # Correction / post-editing options (UI)
    transcription_correction: str = "Brak"
    correction_ollama_model_name: Optional[str] = None
    openrouter_model_name: Optional[str] = None
    # Prompt korekty dla plików tekstowych (TXT/DOCX).
    correction_prompt: Optional[str] = None
    # Osobny prompt korekty dla napisów SRT (segmenty z kodami czasowymi). Gdy pusty,
    # correction_service używa correction_prompt.
    correction_prompt_srt: Optional[str] = None
    transcription_segment_batch_size: int = 200
    translation_ollama_prompt: Optional[str] = None
    translation_openrouter_prompt: Optional[str] = None
    translation_segment_batch_size: int = 250
    ollama_summary_prompt: Optional[str] = None
    summary_gemini_prompt: Optional[str] = None
    summary_openrouter_prompt: Optional[str] = None
    summary_openrouter_model_name: Optional[str] = None
    # Optional custom prompt for BART summarization
    bart_summary_prompt: Optional[str] = None
    # Forced-alignment option: when True, attempt to map corrected text back to timestamps using whisperx
    enable_forced_alignment: bool = False
    # Optional model name to use for whisperx alignment (e.g., 'large-v2')
    forced_alignment_model: Optional[str] = None
    # Show progress on main progress bar while sending segments to Ollama
    show_ollama_progress: bool = True
    # Cooldown (w sekundach) przed wywołaniami Gemini, gdy korekta również korzysta z Gemini
    gemini_cooldown_seconds: float = 4.0
    # MLX (Apple Silicon) model settings
    mlx_model_id: Optional[str] = None
    # llama.cpp (NVIDIA GPU) model settings
    llama_cpp_model_repo: Optional[str] = None
    llama_cpp_model_file: Optional[str] = None
    llama_cpp_n_gpu_layers: int = -1

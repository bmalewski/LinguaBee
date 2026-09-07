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
#   ze źródeł do katalogu projektu;
# - dane użytkownika (wyniki, modele, ustawienia, szablony promptów):
#   w spakowanej aplikacji katalogi standardowe systemu (macOS: ~/Documents/LinguaBee
#   i ~/Library/Application Support/LinguaBee), ze źródeł — jak dotąd obok tego pliku.
import shutil

APP_NAME = "LinguaBee"
_SOURCE_ROOT = os.path.dirname(os.path.abspath(__file__))


def is_frozen() -> bool:
    """True, gdy aplikacja działa jako spakowany plik wykonywalny (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def resource_path(*parts) -> str:
    """Ścieżka do zasobu tylko do odczytu dołączonego do aplikacji."""
    root = getattr(sys, "_MEIPASS", None) if is_frozen() else None
    return os.path.join(root or _SOURCE_ROOT, *parts)


def _user_data_root() -> str:
    """Katalog zapisu dla modeli, ustawień i szablonów promptów."""
    if not is_frozen():
        return _SOURCE_ROOT
    if sys.platform == "darwin":
        return os.path.expanduser(f"~/Library/Application Support/{APP_NAME}")
    if sys.platform.startswith("win"):
        return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), APP_NAME)
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), APP_NAME)


def _user_output_root() -> str:
    """Katalog, do którego trafiają wyniki (TXT/DOCX/SRT, pobrane audio)."""
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
models_dir = os.path.join(data_root, "models")
prompts_dir = os.path.join(data_root, "prompts")
settings_file = os.path.join(data_root, "user_settings.json")
icons_dir = resource_path("icons")
bin_dir = resource_path("bin")  # wbudowane ffmpeg/ffprobe (tylko w spakowanej aplikacji)


def prompts_subdir(*parts) -> str:
    """Zapisywalny katalog szablonów promptów (tworzony na żądanie)."""
    path = os.path.join(prompts_dir, *parts)
    _safe_makedirs(path)
    return path


def _seed_prompts_if_frozen() -> None:
    """Pierwsze uruchomienie spakowanej aplikacji: kopiuje dołączone szablony promptów
    do katalogu użytkownika. Nigdy nie nadpisuje istniejących plików."""
    if not is_frozen():
        return
    src_root = resource_path("prompts")
    if not os.path.isdir(src_root):
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


for _d in (downloads_dir, models_dir, prompts_dir):
    _safe_makedirs(_d)
_seed_prompts_if_frozen()


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
    correction_prompt: Optional[str] = None
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

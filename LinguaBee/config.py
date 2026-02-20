import os
import sys
from dataclasses import dataclass
from typing import Optional, List
import json

# -------------------------------------------------
# Paths
# -------------------------------------------------
# Determine the base path for the application, which works for both normal execution and when bundled with PyInstaller.
base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

# Define the directory where all output files will be saved.
downloads_dir = os.path.join(base_path, "downloads")

# Define the directory where icons are stored.
icons_dir = os.path.join(base_path, "icons")

# Define the directory where all models will be saved.
models_dir = os.path.join(base_path, "models")


# Create the downloads and models directories if they don't already exist.
os.makedirs(downloads_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# File to persist user settings between sessions
settings_file = os.path.join(base_path, "user_settings.json")


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
    force_mono: bool = True
    # Correction / post-editing options (UI)
    transcription_correction: str = "Brak"
    correction_ollama_model_name: Optional[str] = None
    openrouter_model_name: Optional[str] = None
    correction_prompt: Optional[str] = None
    transcription_segment_batch_size: int = 250
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

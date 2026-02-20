from dataclasses import asdict
import torch
from faster_whisper import WhisperModel
from config import TranscriptionConfig, models_dir
from whisper_paragrafizer import paragraphize_segments
from huggingface_utils import login_huggingface

whisper_model_cache = {}


def _is_permission_error(err: Exception) -> bool:
    msg = str(err).lower()
    markers = [
        "403",
        "401",
        "forbidden",
        "unauthorized",
        "permission",
        "uprawnie",
        "access denied",
        "gated",
    ]
    return any(m in msg for m in markers)

class WhisperTranscription:
    def __init__(self, config: TranscriptionConfig, status_signal, progress_signal, finished_signal, is_stopped):
        self.config = config
        self.status_signal = status_signal
        self.progress_signal = progress_signal
        self.finished_signal = finished_signal
        self._is_stopped = is_stopped
        self.segments = None
        self.info = None
        self.text = None

    def transcribe(self, audio_path):
        global whisper_model_cache
        
        model_key = f"{self.config.whisper_variant}_{self.config.whisper_device}_{self.config.whisper_device_index}"
        if model_key not in whisper_model_cache:
            self.status_signal.emit(f"Ładowanie modelu Whisper '{self.config.whisper_variant}' do '{models_dir}' (może potrwać)...", "info")
            if self._is_stopped(): return None, None
            compute_type = "int8" if self.config.whisper_device == "cpu" else "float32"
            # Best-effort login to HF in case model files require authenticated access.
            try:
                login_huggingface()
            except Exception:
                pass
            try:
                whisper_model_cache[model_key] = WhisperModel(
                    self.config.whisper_variant, 
                    device=self.config.whisper_device, 
                    device_index=self.config.whisper_device_index, 
                    compute_type=compute_type,
                    download_root=models_dir
                )
            except Exception as e:
                # Known issue on some environments: turbo fetch can fail with permissions/auth.
                if str(self.config.whisper_variant).strip().lower() == "turbo" and _is_permission_error(e):
                    self.status_signal.emit(
                        "Model 'turbo' zgłosił błąd uprawnień przy pobieraniu. Przełączam awaryjnie na 'large-v3'.",
                        "warning"
                    )
                    fallback_variant = "large-v3"
                    fallback_key = f"{fallback_variant}_{self.config.whisper_device}_{self.config.whisper_device_index}"
                    try:
                        whisper_model_cache[fallback_key] = WhisperModel(
                            fallback_variant,
                            device=self.config.whisper_device,
                            device_index=self.config.whisper_device_index,
                            compute_type=compute_type,
                            download_root=models_dir
                        )
                        model_key = fallback_key
                        self.config.whisper_variant = fallback_variant
                    except Exception as e2:
                        self.finished_signal.emit(
                            f"Błąd podczas ładowania modelu Whisper (turbo i fallback large-v3): {e2}",
                            "error"
                        )
                        return None, None, None
                else:
                    self.finished_signal.emit(
                        f"Błąd podczas ładowania modelu Whisper: {e}. "
                        f"Jeśli to błąd uprawnień, zaloguj Hugging Face tokenem i spróbuj ponownie.",
                        "error"
                    )
                    return None, None, None

        model = whisper_model_cache[model_key]
        device_info = f"{self.config.whisper_device}:{self.config.whisper_device_index}" if self.config.whisper_device == 'cuda' else self.config.whisper_device
        self.status_signal.emit(f"Transkrypcja (Whisper, {self.config.whisper_variant}, {device_info})...", "info")
        if self._is_stopped(): return None, None

        try:
            segments_generator, info = model.transcribe(
                audio_path,
                language=self.config.src_lang_code if self.config.src_lang_code != "auto" else None,
                beam_size=10,
                chunk_length=30, # Długość fragmentu audio w sekundach
                word_timestamps=True,
                patience=2,
                repetition_penalty=1.5,
                no_repeat_ngram_size=10,
                log_prob_threshold=-1.0,
                condition_on_previous_text=False,
                temperature=0,
            )
            
            total_duration = info.duration
            segments = []
            text_list = []
            self.progress_signal.emit(0)
            
            for s in segments_generator:
                if self._is_stopped(): return None, None
                segment_dict = asdict(s)
                segments.append(segment_dict)
                text_list.append(segment_dict["text"])
                if total_duration > 0:
                    progress = (segment_dict["end"] / total_duration) * 100
                    self.progress_signal.emit(int(progress))
            
            if self._is_stopped(): 
                self.text, self.segments, self.info = None, None, None
                return None, None, None
            
            text = "".join(text_list)
            self.progress_signal.emit(100)
            self.status_signal.emit(f"Wykryty język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})", "info")
            
            self.text, self.segments, self.info = text, segments, info
            
            # Optional: run WhisperX alignment to get word-level timestamps and optional diarization
            try:
                use_whisperx = getattr(self.config, "enable_whisperx", False)
            except Exception:
                use_whisperx = False

            # paragraph results (computed later if requested)
            paragraphs = None

            # --- Nowa, główna logika diaryzacji ---
            try:
                use_diarization = getattr(self.config, "enable_diarization", False)
            except AttributeError:
                use_diarization = False

            if use_diarization:
                # Logika diaryzacji została przeniesiona do pliku worker.py,
                # aby była wykonywana po transkrypcji, ale przed zapisem plików.
                # Ten fragment jest celowo pusty.
                pass

            if use_whisperx:
                try:
                    import whisperx
                except Exception as e:
                    self.status_signal.emit(f"WhisperX not available: {e}. Skipping alignment.", "warning")
                    # If paragraphing is enabled, create paragraphs from available segments
                    if getattr(self.config, 'enable_paragraphing', False):
                        try:
                            paragraphs = paragraphize_segments(
                                segments,
                                silence_threshold=getattr(self.config, 'paragraph_silence_threshold', 1.0),
                                max_chars=getattr(self.config, 'paragraph_max_chars', 500),
                                break_on_speaker=getattr(self.config, 'paragraph_break_on_speaker', True),
                                min_sentence_chars=getattr(self.config, 'paragraph_min_sentence_chars', 10),
                            )
                            try:
                                setattr(info, 'paragraphs', paragraphs)
                            except Exception:
                                pass
                            return text, segments, info
                        except Exception:
                            # fallback to original behavior
                            return text, segments, info
                    return text, segments, info

                # determine device for whisperx: prefer explicit config value, fall back to whisper_device
                wx_device = self.config.whisperx_device or self.config.whisper_device
                # normalize device value: allow forms like 'cuda:0' or 'cpu'
                try:
                    if isinstance(wx_device, str) and wx_device.startswith('cuda') and ':' in wx_device:
                        wx_device_name = 'cuda'
                    else:
                        wx_device_name = wx_device
                except Exception:
                    wx_device_name = wx_device

                self.status_signal.emit("Uruchamiam WhisperX alignment (word-level timestamps)...", "info")
                try:
                    # load audio and alignment model
                    self.status_signal.emit("Ładowanie audio za pomocą librosa dla WhisperX...", "info")
                    import librosa
                    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
                    # pad/trim as whisperx examples do
                    try:
                        audio = whisperx.pad_or_trim(audio)
                    except Exception:
                        pass

                    # load align model (language code from detected info)
                    lang = getattr(info, "language", None) or getattr(self.config, "src_lang_code", None) or "en"
                    # Try loading align model on requested device, then fallback to CPU if necessary
                    try:
                        align_model, metadata = whisperx.load_align_model(lang, device=wx_device_name)
                    except Exception:
                        # try alternative signature
                        try:
                            align_model, metadata = whisperx.load_align_model(lang, wx_device_name)
                        except Exception:
                            # fallback to CPU
                            self.status_signal.emit("WhisperX alignment: nie udało się załadować modelu na GPU, próba na CPU...", "warning")
                            align_model, metadata = whisperx.load_align_model(lang, device='cpu')

                    # whisperx.align expects the segments format from a prior transcription
                    # Try alignment; prefer device name, but some API variants accept different args
                    try:
                        word_segments = whisperx.align(segments, align_model, metadata, audio, wx_device_name)
                    except TypeError:
                        try:
                            word_segments = whisperx.align(segments, align_model, metadata, audio)
                        except TypeError:
                            try:
                                fw_model = model
                            except Exception:
                                fw_model = None
                            word_segments = whisperx.align(segments, align_model, metadata, audio, fw_model, wx_device_name)

                    # word_segments should be a list of dicts; replace segments with word-level segments
                    segments = word_segments or segments
                    self.status_signal.emit("WhisperX alignment zakończony.", "info")

                    # Optional diarization merge
                    # Stara logika diaryzacji w whisperx została zastąpiona nowym, głównym mechanizmem powyżej.
                    # Pozostawiamy tę sekcję pustą, aby uniknąć konfliktów.
                    pass

                except Exception as e:
                    self.status_signal.emit(f"WhisperX alignment failed: {e}", "warning")

            # If paragraphing requested, compute paragraphs (if not already computed) and attach to info
            if getattr(self.config, 'enable_paragraphing', False):
                try:
                    if paragraphs is None:
                        paragraphs = paragraphize_segments(
                            segments,
                            silence_threshold=getattr(self.config, 'paragraph_silence_threshold', 1.0),
                            max_chars=getattr(self.config, 'paragraph_max_chars', 500),
                            break_on_speaker=getattr(self.config, 'paragraph_break_on_speaker', True),
                            min_sentence_chars=getattr(self.config, 'paragraph_min_sentence_chars', 10),
                        )
                    try:
                        setattr(info, 'paragraphs', paragraphs)
                    except Exception:
                        pass
                    return text, segments, info
                except Exception:
                    # on failure, fall back to returning the original three-tuple
                    return text, segments, info

            return text, segments, info

        except Exception as e:
            self.finished_signal.emit(f"Błąd podczas transkrypcji Whisper: {e}", "error")
            return None, None, None
    
    def paragraphize(self):
        """Computes paragraphs from the existing transcription results."""
        if not self.segments or not self.info:
            self.status_signal.emit("Brak segmentów do podziału na akapity. Pomiń.", "warning")
            return

        self.status_signal.emit("Dzielenie transkryptu na akapity...", "info")
        try:
            paragraphs = paragraphize_segments(
                self.segments,
                silence_threshold=getattr(self.config, 'paragraph_silence_threshold', 1.0),
                max_chars=getattr(self.config, 'paragraph_max_chars', 500),
                break_on_speaker=getattr(self.config, 'paragraph_break_on_speaker', True),
                min_sentence_chars=getattr(self.config, 'paragraph_min_sentence_chars', 10),
            )
            setattr(self.info, 'paragraphs', paragraphs)
            self.status_signal.emit("Podział na akapity zakończony.", "info")
        except Exception as e:
            self.status_signal.emit(f"Nie udało się podzielić tekstu na akapity: {e}", "warning")

def release_whisper_model():
    global whisper_model_cache
    for key in list(whisper_model_cache.keys()):
        model = whisper_model_cache.pop(key)
        del model
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

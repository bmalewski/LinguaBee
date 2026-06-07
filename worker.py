import os

from PySide6.QtCore import QThread, Signal

from config import TranscriptionConfig, downloads_dir
from downloader import download_audio
from file_utils import is_video_file, extract_audio_from_video
from whisper_transcription import WhisperTranscription, release_whisper_model
from translation_manager import translate
from summarization_manager import summarize
from text_utils import format_transcript, add_missing_spaces
from ollama_refiner import OllamaRefiner
from correction_service import run_correction_step, _redact_api_key_in_message
from correction_adapters import CorrectionAdapters
from artifact_io_service import read_artifact, write_artifact
from types import SimpleNamespace
import re
import time
from whisper_paragrafizer import paragraphs_to_plaintext
try:
    from whisper_aligner import forced_align_refined_text
except (ImportError, ModuleNotFoundError):
    forced_align_refined_text = None

# Global cache for models
nllb_translator_cache = {}
helsinki_translator_cache = {}

class TranscriptionThread(QThread):
    progress_signal = Signal(int)
    status_signal = Signal(str, str)
    preview_signal = Signal(str, str, str)
    finished_signal = Signal(str, str)

    def __init__(self, config: TranscriptionConfig):
        super().__init__()
        self.config = config
        self.audio_path_to_delete = None
        self._is_stopped = False
        # Przekazanie globalnych cache do instancji
        global nllb_translator_cache
        global helsinki_translator_cache

    def stop(self):
        self._is_stopped = True

    def _emit_preview(self, section: str, source_name: str, content: str):
        try:
            if not isinstance(content, str):
                content = str(content or "")
            payload = content.strip()
            if not payload:
                return
            max_chars = 250000
            if len(payload) > max_chars:
                payload = payload[:max_chars] + "\n\n[... obcięto podgląd ...]"
            self.preview_signal.emit(section, str(source_name or "Wynik"), payload)
        except Exception:
            pass

    def _format_srt_timestamp(self, seconds) -> str:
        try:
            val = float(seconds or 0.0)
        except Exception:
            val = 0.0
        if val < 0:
            val = 0.0
        total_ms = int(round(val * 1000.0))
        hours = total_ms // 3600000
        minutes = (total_ms % 3600000) // 60000
        secs = (total_ms % 60000) // 1000
        millis = total_ms % 1000
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def _segments_to_srt_text(self, segments) -> str:
        if not segments:
            return ""
        blocks = []
        for idx, seg in enumerate(segments, start=1):
            try:
                start_ts = self._format_srt_timestamp(seg.get("start", 0.0))
                end_ts = self._format_srt_timestamp(seg.get("end", seg.get("start", 0.0)))
                txt = str(seg.get("text", "")).strip()
            except Exception:
                start_ts = "00:00:00,000"
                end_ts = "00:00:00,000"
                txt = ""
            if not txt:
                continue
            blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{txt}")
        return "\n\n".join(blocks).strip()

    def _build_srt_preview(self, segments, fallback_text: str = "") -> str:
        srt_text = self._segments_to_srt_text(segments)
        if srt_text:
            return srt_text
        fallback = str(fallback_text or "").strip()
        if not fallback:
            return ""
        return f"1\n00:00:00,000 --> 00:00:00,000\n{fallback}"

    def run(self):
        if self._is_stopped:
            self.finished_signal.emit("Proces zatrzymany przez użytkownika.", "info")
            return

        # Ustawienie cache dla managera tłumaczeń
        import translation_manager
        translation_manager.extern_nllb_translator_cache = nllb_translator_cache
        translation_manager.extern_helsinki_translator_cache = helsinki_translator_cache

        global TORCH_AVAILABLE, CUDA_AVAILABLE, FASTER_WHISPER_AVAILABLE, torch
        try:
            import torch
            TORCH_AVAILABLE = True
            CUDA_AVAILABLE = torch.cuda.is_available()
            import faster_whisper
            FASTER_WHISPER_AVAILABLE = True
        except ImportError as e:
            TORCH_AVAILABLE = False
            CUDA_AVAILABLE = False
            FASTER_WHISPER_AVAILABLE = False
            if self.config.transcription_model == "Whisper (lokalny)":
                self.finished_signal.emit(f"Błąd importu biblioteki: {e}. Upewnij się, że wszystkie zależności są zainstalowane (torch, faster-whisper).", "error")
                return
        
        try:
            files_to_process = self.config.local_files.copy()
            if self.config.url:
                files_to_process.insert(0, self.config.url)

            is_multi_file = len(files_to_process) > 1

            log_message = f"Wybrano konfigurację: "
            log_message += f"Transkrypcja: {self.config.transcription_model} "
            if self.config.transcription_model == "Whisper (lokalny)":
                device_info = f"{self.config.whisper_device}:{self.config.whisper_device_index}" if self.config.whisper_device == 'cuda' else self.config.whisper_device
                log_message += f"(wariant: {self.config.whisper_variant}, urządzenie: {device_info}) "
            log_message += f"Język źródłowy: {self.config.src_lang_code}. "

            if self.config.translation_model != "Brak":
                log_message += f"Tłumaczenie: {self.config.translation_model} "
                if self.config.translation_model == "NLLB (lokalny)":
                    device_info = f"{self.config.nllb_device}:{self.config.nllb_device_index}" if self.config.nllb_device == 'cuda' else self.config.nllb_device
                    log_message += f"(wariant: {self.config.nllb_variant}, urządzenie: {device_info}) "
                elif self.config.translation_model == "Ollama (lokalny)":
                    log_message += f"(model: {self.config.ollama_model_name}) "
                log_message += f"Język docelowy: {self.config.tgt_lang_code}. "
                if self.config.translation_src_lang_code != "auto":
                    log_message += f"Język źródłowy tłumaczenia: {self.config.translation_src_lang_code}. "
            else:
                log_message += "Tłumaczenie: Brak. "
            self.status_signal.emit(log_message, "info")

            gemini_rate_limited_until = 0.0

            for i, file_or_url in enumerate(files_to_process):
                if self._is_stopped:
                    break
                
                try:
                    self.audio_path_to_delete = None
                    is_srt_input = False
                    is_text_input = False
                    segments = []
                    text = ""
                    info = None
                    translated_text = None
                    translated_segments = None
                    
                    if is_multi_file:
                        self.status_signal.emit(f"--- Plik {i+1}/{len(files_to_process)} ---", "info")

                    if file_or_url.startswith("http"):
                        if self.config.transcription_model == "Brak":
                            self.status_signal.emit(
                                "Pominięto URL: wybrano model transkrypcji 'Brak'. Dla URL wymagany jest Whisper.",
                                "warning"
                            )
                            continue
                        self.status_signal.emit(f"Pobieranie audio z {file_or_url}...", "info")
                        audio_path, base_name = download_audio(file_or_url, self.progress_signal, self.status_signal, self.finished_signal, lambda: self._is_stopped)
                        if audio_path is None:
                            self.status_signal.emit(f"Pominięto {file_or_url} z powodu błędu pobierania.", "warning")
                            continue
                        self.audio_path_to_delete = audio_path
                    else:
                        local_path = file_or_url
                        base_name = os.path.splitext(os.path.basename(local_path))[0]
                        self.status_signal.emit(f"Przetwarzanie pliku: {local_path}", "info")

                        if local_path.lower().endswith('.srt'):
                            is_srt_input = True
                            try:
                                artifact = read_artifact(local_path, "srt")
                                text, segments = artifact.text, (artifact.segments or [])
                                guessed_lang = self.config.translation_src_lang_code if self.config.translation_src_lang_code != 'auto' else self.config.src_lang_code
                                if guessed_lang == 'auto':
                                    guessed_lang = 'it'
                                info = SimpleNamespace(language=guessed_lang, paragraphs=None)
                                audio_path = None
                                self.status_signal.emit(f"Wczytano plik SRT: {os.path.basename(local_path)} (segmentów: {len(segments)})", "info")
                            except Exception as e:
                                self.status_signal.emit(f"Nie udało się wczytać pliku SRT: {e}", "error")
                                continue
                        elif local_path.lower().endswith(('.txt', '.docx')):
                            is_text_input = True
                            try:
                                ext = os.path.splitext(local_path)[1].lower().lstrip('.')
                                artifact = read_artifact(local_path, ext)
                                text, segments = artifact.text, (artifact.segments or [])
                                guessed_lang = self.config.translation_src_lang_code if self.config.translation_src_lang_code != 'auto' else self.config.src_lang_code
                                if guessed_lang == 'auto':
                                    guessed_lang = 'pl'
                                info = SimpleNamespace(language=guessed_lang, paragraphs=None)
                                audio_path = None
                                self.status_signal.emit(
                                    f"Wczytano plik tekstowy: {os.path.basename(local_path)} (znaków: {len(text or '')})",
                                    "info"
                                )
                                if not text or not str(text).strip():
                                    self.status_signal.emit(f"Pominięto pusty plik tekstowy: {os.path.basename(local_path)}", "warning")
                                    continue
                            except Exception as e:
                                self.status_signal.emit(f"Nie udało się wczytać pliku tekstowego: {e}", "error")
                                continue
                        else:
                            if self.config.transcription_model == "Brak":
                                self.status_signal.emit(
                                    f"Pominięto plik {os.path.basename(local_path)}: model transkrypcji ustawiony na 'Brak' obsługuje tylko wejścia tekstowe (TXT/DOCX/SRT).",
                                    "warning"
                                )
                                continue
                            if is_video_file(local_path):
                                extracted_audio_path = extract_audio_from_video(local_path, self.status_signal, self.progress_signal)
                                if not extracted_audio_path:
                                    self.status_signal.emit(f"Pominięto plik wideo z powodu błędu ekstrakcji audio: {os.path.basename(local_path)}", "warning")
                                    continue
                                audio_path = extracted_audio_path
                                self.audio_path_to_delete = extracted_audio_path
                            else:
                                audio_path = local_path

                    # Opcjonalne przetwarzanie audio (odszumianie, normalizacja, mono)
                    # Uruchamiamy przetwarzanie w izolowanym subprocessie, aby uniknąć
                    # deadlocków / crashy natywnych bibliotek (torch/pyannote) na Windows.
                    _needs_audio_proc = any([
                        getattr(self.config, 'enable_denoising', False),
                        getattr(self.config, 'enable_normalization', False),
                        getattr(self.config, 'force_mono', False),
                    ])
                    if not is_srt_input and not is_text_input and _needs_audio_proc:
                        try:
                            import subprocess, json, sys

                            runner_path = os.path.join(os.path.dirname(__file__), 'tools', 'process_audio_runner.py')
                            timeout_secs = getattr(self.config, 'audio_processing_timeout', 300)
                            cmd = [sys.executable, runner_path, audio_path]
                            # Build config to pass to runner (respect GUI settings)
                            cfg_payload = {
                                'enable_denoising': bool(getattr(self.config, 'enable_denoising', False)),
                                'enable_normalization': bool(getattr(self.config, 'enable_normalization', False)),
                                'force_mono': bool(getattr(self.config, 'force_mono', False)),
                            }
                            self.status_signal.emit(f"Uruchamiam proces przetwarzania audio (izolowany): {os.path.basename(audio_path)}", "info")
                            # Run without text decoding so we can decode stdout as UTF-8 safely
                            proc = subprocess.run(cmd, input=json.dumps(cfg_payload).encode('utf-8'), capture_output=True, text=False, timeout=timeout_secs)

                            if proc.returncode == 0:
                                try:
                                    # proc.stdout is bytes; decode as UTF-8 (runner writes UTF-8)
                                    stdout_bytes = proc.stdout or b''
                                    result = json.loads(stdout_bytes.decode('utf-8'))
                                    processed_path = result.get('processed_path', audio_path)
                                    # Jeśli runner zwrócił nową ścieżkę, użyj jej
                                    if processed_path and processed_path != audio_path:
                                        self.audio_path_to_delete = processed_path
                                        audio_path = processed_path
                                        self.status_signal.emit(f"Przetwarzanie audio (subprocess) zakończone: {processed_path}", "info")
                                    else:
                                        self.status_signal.emit("Przetwarzanie audio (subprocess) zakończone: brak zmiany pliku.", "info")
                                except Exception as e:
                                    self.status_signal.emit(f"Nieprawidłowy wynik z procesu przetwarzania audio: {e}. Wyjście: {proc.stdout}", "warning")
                            else:
                                # Zapisz stdout/stderr dla diagnostyki, użyjemy oryginalnego pliku jako fallback
                                self.status_signal.emit(f"Proces przetwarzania audio zakończył się kodem {proc.returncode}. stderr: {proc.stderr}", "warning")
                                processed_path = audio_path
                        except subprocess.TimeoutExpired:
                            self.status_signal.emit("Proces przetwarzania audio przekroczył limit czasu i został przerwany.", "warning")
                            processed_path = audio_path
                        except Exception as e:
                            self.status_signal.emit(f"Błąd podczas uruchamiania procesu przetwarzania audio: {e}", "warning")
                            processed_path = audio_path

                    if is_srt_input:
                        # `text`, `segments`, `info` zostały już przygotowane podczas wczytywania SRT.
                        self.status_signal.emit("Pominięto transkrypcję audio: używam bezpośrednio danych z pliku SRT.", "info")
                    elif is_text_input:
                        # `text` i `info` zostały już przygotowane podczas wczytywania pliku tekstowego.
                        self.status_signal.emit("Pominięto transkrypcję audio: używam bezpośrednio treści pliku tekstowego.", "info")

                    # Helper: create a status callback wrapper that also emits progress_signal
                    # This is defined here so it's always available regardless of formatting branch.
                    def _make_ollama_status_cb(base_cb, show_progress=True):
                        prog_re = re.compile(r"Refinowanie fragmentu\s*(\d+)/(\d+)")
                        def _cb(msg, level="info"):
                            try:
                                base_cb(msg, level)
                            except Exception:
                                pass
                            if show_progress:
                                try:
                                    m = prog_re.search(msg)
                                    if m:
                                        num = int(m.group(1))
                                        total = int(m.group(2))
                                        pct = int((num / total) * 100)
                                        try:
                                            self.progress_signal.emit(pct)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                        return _cb

                    if self.config.transcription_model == "Whisper (lokalny)" and not is_srt_input and not is_text_input:
                        if not FASTER_WHISPER_AVAILABLE:
                            self.finished_signal.emit("Błąd: Brak biblioteki 'faster-whisper'.\nZainstaluj ją, wpisując: pip install faster-whisper", "error")
                            return
                        
                        whisper_transcriber = WhisperTranscription(
                            self.config, 
                            self.status_signal, 
                            self.progress_signal, 
                            self.finished_signal, 
                            lambda: self._is_stopped
                        )
                        text, segments, info = whisper_transcriber.transcribe(audio_path)

                        if text is None: # Błąd wystąpił w transkrypcji
                            continue

                        # New integrated diarization and paragraphing workflow
                        if self.config.enable_diarization and segments:
                            try:
                                self.status_signal.emit("Uruchamiam diaryzację (rozpoznawanie mówców)...", "info")
                                from audio_processing import diarize_audio, assign_speakers_to_words, create_speaker_paragraphs
                                diarization_timeline = diarize_audio(audio_path, self.config, self.status_signal, self.progress_signal)
                                
                                if diarization_timeline:
                                    self.status_signal.emit("Integrowanie wyników diaryzacji z transkrypcją...", "info")
                                    
                                    # Assign speakers to individual words
                                    segments_with_word_speakers = assign_speakers_to_words(diarization_timeline, segments)
                                    
                                    # Create new paragraphs based on speaker turns
                                    speaker_paragraphs = create_speaker_paragraphs(
                                        segments_with_word_speakers,
                                        max_chars=getattr(self.config, 'paragraph_max_chars', 500),
                                        break_on_speaker=True
                                    )
                                    
                                    # Attach the new speaker-aware paragraphs to the info object for saving
                                    if speaker_paragraphs:
                                        setattr(info, 'paragraphs', speaker_paragraphs)
                                        self.status_signal.emit("Utworzono nowe akapity w oparciu o zmiany mówców.", "info")

                                    # Also, update the main 'segments' list to contain speaker info for SRT output.
                                    for seg in segments:
                                        seg_start, seg_end = seg.get('start', 0.0), seg.get('end', 0.0)
                                        best_speaker, max_overlap = 'UNKNOWN', 0.0
                                        for turn in diarization_timeline:
                                            overlap = max(0, min(seg_end, turn['end']) - max(seg_start, turn['start']))
                                            if overlap > max_overlap:
                                                max_overlap, best_speaker = overlap, turn['speaker']
                                        seg['speaker'] = best_speaker
                            except Exception as e:
                                self.status_signal.emit(f"Diaryzacja nie powiodła się: {e}. Kontynuowanie bez informacji o mówcach.", "warning")
                        
                        # Fallback to old paragraphing if diarization is disabled but paragraphing is enabled
                        elif getattr(self.config, 'enable_paragraphing', False):
                            self.status_signal.emit("Uruchamiam domyślny podział na akapity (bez diaryzacji)...", "info")
                            try:
                                whisper_transcriber.paragraphize()
                            except Exception as e:
                                self.status_signal.emit(f"Błąd podczas dzielenia na akapity: {e}", "warning")

                    # OpenAI transcription option removed — only local Whisper is supported now.

                    if self._is_stopped: break

                    if self.config.formats_original and text:
                        self.status_signal.emit("Zapisywanie oryginalnych plików transkrypcji...", "info")
                        if self._is_stopped: break
                        # If user requested formatting, produce a formatted version
                        use_formatted = getattr(self.config, 'format_model', 'Brak') != 'Brak'
                        formatted_text = None
                        if use_formatted:
                            try:
                                title = base_name
                                formatted_text = format_transcript(title, text, segments)
                            except Exception:
                                formatted_text = None

                            # If formatting mode involves Ollama, try a lightweight refinement step using the selected Ollama model
                            try:
                                fmt_mode = getattr(self.config, 'format_model', 'Brak')
                                fmt_ollama_model = getattr(self.config, 'ollama_format_model', '')
                                # Case-insensitive check for Ollama in the formatting mode
                                if formatted_text and fmt_mode and ('ollama' in fmt_mode.lower()) and fmt_ollama_model:
                                    # Fix missing spaces/punctuation before sending to Ollama
                                    try:
                                        formatted_text = add_missing_spaces(formatted_text)
                                    except Exception:
                                        pass
                                    # Reuse the status callback factory defined above in run()
                                    refiner = OllamaRefiner(fmt_ollama_model, status_callback=_make_ollama_status_cb(self.status_signal.emit, True))
                                    try:
                                        try:
                                            self.status_signal.emit(f"Wysyłam sformatowany tekst do Ollama (model: {fmt_ollama_model}) w celu refinowania...", "info")
                                        except Exception:
                                            pass
                                        refined = refiner.refine(formatted_text)
                                        if refined and refined.strip():
                                            formatted_text = refined
                                            self.status_signal.emit("Zastosowano refinowanie przez lokalny model Ollama.", "info")
                                    except Exception:
                                        self.status_signal.emit("Refinement przez Ollama nie powiódł się; używam heurystycznego formatowania.", "warning")
                            except Exception:
                                # ignore formatting-related failures and continue to correction step
                                pass

                        # --- Correction / post-editing step ---
                        try:
                            # If paragraphing was requested for this job and paragraphs were computed,
                            # overwrite the original TXT/DOCX outputs with paragraph text instead
                            paras = getattr(info, 'paragraphs', None)
                            use_paragraphs = bool(paras) and getattr(self.config, 'enable_paragraphing', False)

                            for fmt in self.config.formats_original:
                                ext = fmt.lower()
                                path = os.path.join(downloads_dir, f"{base_name}_original.{ext}")
                                try:
                                    if ext == "txt":
                                        if use_paragraphs:
                                            write_artifact(text=paragraphs_to_plaintext(paras), segments=None, path=path, ext=ext)
                                        else:
                                            write_artifact(text=formatted_text if formatted_text is not None else text, segments=None, path=path, ext=ext)
                                    elif ext == "docx":
                                        if use_paragraphs:
                                            write_artifact(text=paragraphs_to_plaintext(paras), segments=None, path=path, ext=ext)
                                        else:
                                            write_artifact(text=formatted_text if formatted_text is not None else text, segments=None, path=path, ext=ext)
                                    elif ext == "srt" and segments:
                                        write_artifact(
                                            text="",
                                            segments=segments,
                                            path=path,
                                            ext=ext,
                                            srt_max_lines=getattr(self.config, 'srt_max_lines', 2),
                                            srt_max_chars_per_line=getattr(self.config, 'srt_max_chars_per_line', 25),
                                        )
                                except Exception as e:
                                    # don't fail the whole job if a particular format fails
                                    self.status_signal.emit(f"Błąd zapisu formatu {ext}: {e}", "warning")

                            # Zwolnij zasoby Whisper (w tym VRAM) przed uruchomieniem korekty.
                            try:
                                self.status_signal.emit("Zwalnianie pamięci VRAM po transkrypcji i przed korektą...", "info")
                                release_whisper_model()
                                import gc
                                gc.collect()
                                if CUDA_AVAILABLE:
                                    torch.cuda.empty_cache()
                            except Exception as e:
                                self.status_signal.emit(f"Nie udało się w pełni zwolnić VRAM przed korektą: {e}", "warning")

                            # --- Correction / post-editing step (AFTER files are saved) ---
                            try:
                                correction_adapters = CorrectionAdapters(
                                    config=self.config,
                                    status_cb=self.status_signal.emit,
                                    progress_cb=self.progress_signal.emit,
                                    make_ollama_status_cb=_make_ollama_status_cb,
                                )
                                text, segments, gemini_rate_limited_until = run_correction_step(
                                    config=self.config,
                                    base_name=base_name,
                                    file_or_url=file_or_url,
                                    local_path=local_path if 'local_path' in locals() else "",
                                    text=text,
                                    segments=segments,
                                    gemini_rate_limited_until=gemini_rate_limited_until,
                                    status_cb=self.status_signal.emit,
                                    adapters=correction_adapters,
                                )
                            except Exception as e:
                                self.status_signal.emit(f"Błąd krytyczny kroku korekty: {e}", "warning")
                        except Exception as e:
                            gem_key = getattr(self.config, 'gemini_key', None)
                            self.status_signal.emit(
                                f"Wystąpił błąd podczas przetwarzania {file_or_url}: {_redact_api_key_in_message(str(e), gem_key)}",
                                "error"
                            )

                    if self.config.translation_model != "Brak":
                        release_whisper_model()
                        if self._is_stopped: break
                        translated_text, translated_segments = translate(self.config, text, segments, info, self.status_signal, self.progress_signal, self.finished_signal, lambda: self._is_stopped)

                    try:
                        transcription_preview_srt = self._build_srt_preview(segments, text)
                        if transcription_preview_srt:
                            self._emit_preview("transcription", base_name, transcription_preview_srt)
                    except Exception:
                        pass

                    try:
                        translation_preview_srt = self._build_srt_preview(translated_segments, translated_text)
                        if translation_preview_srt:
                            self._emit_preview("translation", base_name, translation_preview_srt)
                    except Exception:
                        pass

                    if self._is_stopped: break

                    if self.config.formats_translated and (translated_text is not None or translated_segments is not None):
                        self.status_signal.emit("Zapisywanie przetłumaczonych plików...", "info")
                        if self._is_stopped: break
                        for fmt in self.config.formats_translated:
                            ext = fmt.lower()
                            path = os.path.join(downloads_dir, f"{base_name}_translation.{ext}")
                            if ext in ["txt", "docx"] and translated_text is not None:
                                if ext == "txt":
                                    write_artifact(text=translated_text, segments=None, path=path, ext=ext)
                                elif ext == "docx":
                                    write_artifact(text=translated_text, segments=None, path=path, ext=ext)
                            elif ext == "srt" and translated_segments is not None:
                                write_artifact(text="", segments=translated_segments, path=path, ext=ext)

                    if self._is_stopped: break

                    if self.config.summary_model != "Brak" and text:
                        summary_text = None
                        try:
                            corr_mode_now = str(getattr(self.config, 'transcription_correction', 'Brak') or '').lower()
                            if self.config.summary_model == "Gemini":
                                now_ts = time.time()
                                if now_ts < gemini_rate_limited_until:
                                    wait_left = int(max(1, gemini_rate_limited_until - now_ts))
                                    wait_for = min(wait_left, 180)
                                    self.status_signal.emit(
                                        f"Aktywny cooldown Gemini po 429 (~{wait_left}s). Czekam {wait_for}s i ponawiam streszczenie...",
                                        "warning"
                                    )
                                    time.sleep(wait_for)
                                    if not self._is_stopped:
                                        summary_text = summarize(self.config, text, info, self.status_signal, self.progress_signal, self.finished_signal, lambda: self._is_stopped)
                                else:
                                    if 'gemini' in corr_mode_now:
                                        cooldown_s = float(getattr(self.config, 'gemini_cooldown_seconds', 4.0) or 4.0)
                                        cooldown_s = max(1.0, min(cooldown_s, 15.0))
                                        self.status_signal.emit(f"Cooldown przed streszczeniem Gemini: {cooldown_s:.1f}s", "info")
                                        time.sleep(cooldown_s)
                                    summary_text = summarize(self.config, text, info, self.status_signal, self.progress_signal, self.finished_signal, lambda: self._is_stopped)
                            else:
                                summary_text = summarize(self.config, text, info, self.status_signal, self.progress_signal, self.finished_signal, lambda: self._is_stopped)
                        except Exception:
                            summary_text = summarize(self.config, text, info, self.status_signal, self.progress_signal, self.finished_signal, lambda: self._is_stopped)
                        if summary_text:
                            try:
                                self._emit_preview("summary", base_name, summary_text)
                            except Exception:
                                pass
                            self.status_signal.emit("Zapisywanie plików streszczenia...", "info")
                            for fmt in self.config.formats_summary:
                                ext = fmt.lower()
                                path = os.path.join(downloads_dir, f"{base_name}_summary.{ext}")
                                if ext == "txt":
                                    write_artifact(text=summary_text, segments=None, path=path, ext=ext)
                                elif ext == "docx":
                                    write_artifact(text=summary_text, segments=None, path=path, ext=ext)
                    
                    if is_multi_file:
                        finished_name = os.path.basename(audio_path) if audio_path else os.path.basename(str(file_or_url))
                        self.status_signal.emit(f"Zakończono: {finished_name}", "success")
                    
                except Exception as e:
                    gem_key = getattr(self.config, 'gemini_key', None)
                    self.status_signal.emit(
                        f"Wystąpił błąd podczas przetwarzania {file_or_url}: {_redact_api_key_in_message(str(e), gem_key)}",
                        "error"
                    )
                    continue
                finally:
                    if self.audio_path_to_delete and os.path.exists(self.audio_path_to_delete):
                        if self.config.delete_audio:
                            os.remove(self.audio_path_to_delete)
                            self.status_signal.emit(f"Usunięto pobrany plik audio: {self.audio_path_to_delete}", "info")
                        else:
                            self.status_signal.emit(f"Pobrany plik audio pozostaje w: {self.audio_path_to_delete}", "info")
        finally:
            self.status_signal.emit("Końcowe czyszczenie zasobów...", "info")
            try:
                for key in list(nllb_translator_cache.keys()):
                    translator = nllb_translator_cache.pop(key)
                    if hasattr(translator, 'release'):
                        translator.release()
                    del translator
                
                for key in list(helsinki_translator_cache.keys()):
                    translator = helsinki_translator_cache.pop(key)
                    if hasattr(translator, 'release'):
                        translator.release()
                    del translator

                release_whisper_model()
                
                import gc
                gc.collect()
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                self.status_signal.emit("Zasoby zwolnione.", "info")
            except Exception as e:
                self.status_signal.emit(f"Błąd podczas końcowego czyszczenia: {e}", "error")

            if self._is_stopped:
                self.finished_signal.emit("Proces zatrzymany przez użytkownika.", "info")
            else:
                self.finished_signal.emit(f"Zakończono wszystkie zadania. Pliki zapisane w {downloads_dir}", "success")

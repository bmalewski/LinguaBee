import os

from PySide6.QtCore import QThread, Signal

from config import TranscriptionConfig, downloads_dir
from downloader import download_audio
from file_utils import save_txt, save_docx, save_html, save_srt, is_video_file, extract_audio_from_video, load_srt
from whisper_transcription import WhisperTranscription, release_whisper_model
from translation_manager import translate
from summarization_manager import summarize
from text_utils import format_transcript, add_missing_spaces, redistribute_text_to_segments
from ollama_refiner import OllamaRefiner
from types import SimpleNamespace
import re
import time
from whisper_paragrafizer import paragraphs_to_plaintext
import httpx
try:
    from whisper_aligner import forced_align_refined_text
except Exception:
    forced_align_refined_text = None


def _send_to_gemini(api_key: str, prompt: str, input_text: str, model: str = "gemini-2.5-flash") -> str:
    """Send prompt+input to Gemini-style Generative API via REST (best-effort).

    This uses the Google Generative Language REST stub for simple deployments.
    If the environment has a different Gemini endpoint, this function may need adjustment.
    """
    if not api_key:
        return ""

    normalized_model = model.strip() if isinstance(model, str) and model.strip() else "gemini-2.5-flash"
    if normalized_model.startswith("models/"):
        normalized_model = normalized_model.split("/", 1)[1]

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt.strip() + "\n\nTekst do poprawy:\n" + input_text.strip()
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 4096,
        }
    }

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{normalized_model}:generateContent"
    last_error = None
    with httpx.Client(timeout=90) as c:
        for attempt in range(4):
            try:
                r = c.post(endpoint + f"?key={api_key}", json=payload, headers={"Content-Type": "application/json"})
                if r.status_code in (429, 503) and attempt < 3:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        wait_s = float(retry_after) if retry_after else float(2 ** attempt)
                    except Exception:
                        wait_s = float(2 ** attempt)
                    time.sleep(min(wait_s, 10.0))
                    continue
                r.raise_for_status()
                j = r.json()
                raw = {'endpoint': endpoint + '?key=REDACTED', 'status': r.status_code, 'response': j}
                _save_raw_gemini_response(raw)

                text_parts = []

                def _collect_text(node):
                    if isinstance(node, dict):
                        t = node.get("text")
                        if isinstance(t, str) and t.strip():
                            text_parts.append(t.strip())
                        for v in node.values():
                            _collect_text(v)
                    elif isinstance(node, list):
                        for it in node:
                            _collect_text(it)

                _collect_text(j.get("candidates") if isinstance(j, dict) else j)
                if text_parts:
                    return "\n".join(text_parts).strip()
                return ""
            except Exception as e:
                last_error = e

    if last_error is not None:
        raise last_error
    return ""


def _send_to_openrouter(api_key: str, prompt: str, input_text: str, model: str = "google/gemini-2.5-flash") -> str:
    """Send prompt+input to OpenRouter chat completions endpoint."""
    if not api_key:
        return ""

    normalized_model = model.strip() if isinstance(model, str) and model.strip() else "google/gemini-2.5-flash"
    payload = {
        "model": normalized_model,
        "messages": [
            {"role": "system", "content": prompt.strip()},
            {"role": "user", "content": input_text.strip()},
        ],
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://linguabee.local",
        "X-Title": "LinguaBee",
    }

    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    last_error = None
    with httpx.Client(timeout=90) as c:
        for attempt in range(4):
            try:
                r = c.post(endpoint, json=payload, headers=headers)
                if r.status_code in (429, 503) and attempt < 3:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        wait_s = float(retry_after) if retry_after else float(2 ** attempt)
                    except Exception:
                        wait_s = float(2 ** attempt)
                    time.sleep(min(wait_s, 10.0))
                    continue
                r.raise_for_status()
                j = r.json()
                try:
                    choices = j.get("choices") if isinstance(j, dict) else None
                    if choices and isinstance(choices, list):
                        msg = choices[0].get("message", {})
                        txt = msg.get("content", "")
                        if isinstance(txt, str):
                            return txt.strip()
                except Exception:
                    pass
                return ""
            except Exception as e:
                last_error = e

    if last_error is not None:
        raise last_error
    return ""


def _save_raw_gemini_response(obj: dict):
    try:
        import json
        out_dir = os.path.join(downloads_dir, 'corrected')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, 'raw_gemini_response.json')
        with open(out_file, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _try_parse_json_array(response_text: str):
    txt = (response_text or "").strip()
    if not txt:
        return None

    # Remove markdown code fences if present
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```$", "", txt)

    try:
        import json as _json
        parsed = _json.loads(txt)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Sometimes models return Python-like list with single quotes.
    try:
        import ast as _ast
        parsed = _ast.literal_eval(txt)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    try:
        candidates = re.findall(r"(\[[\s\S]*?\])", txt, re.DOTALL)
        for cand in candidates:
            import json as _json
            try:
                parsed = _json.loads(cand)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                try:
                    import ast as _ast
                    parsed = _ast.literal_eval(cand)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _parse_jsonish_list_lines(response_text: str):
    """Best-effort parser for outputs like:
    [
      "line1",
      "line2",
    ]
    """
    txt = (response_text or "").strip()
    if not txt:
        return None

    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```$", "", txt)

    out = []
    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in ("[", "]"):
            continue
        if line.startswith("["):
            line = line[1:].lstrip()
        if line.endswith("]"):
            line = line[:-1].rstrip()
        if line.endswith(","):
            line = line[:-1].rstrip()

        # Strip matching quotes
        if len(line) >= 2 and ((line[0] == '"' and line[-1] == '"') or (line[0] == "'" and line[-1] == "'")):
            line = line[1:-1]

        line = line.replace('\\"', '"').replace("\\'", "'").strip()
        if line:
            out.append(line)

    return out if out else None


def _extract_saved_text(path: str, ext: str):
    ext = (ext or "").lower()
    if ext == "txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), None
    if ext == "docx":
        from docx import Document
        doc = Document(path)
        txt = "\n\n".join([p.text for p in doc.paragraphs if p.text and p.text.strip()])
        return txt, None
    if ext == "html":
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        txt = re.sub(r"<[^>]+>", " ", html)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt, None
    if ext == "srt":
        txt, segs = load_srt(path)
        return txt, segs
    return "", None


def _redact_api_key_in_message(msg: str, api_key: str) -> str:
    try:
        if isinstance(api_key, str) and api_key:
            return str(msg).replace(api_key, "REDACTED")
    except Exception:
        pass
    return str(msg)


def _chunk_srt_segments(segments, max_items: int = 200, max_chars: int = 2000):
    chunks = []
    current = []
    current_chars = 0
    for seg in segments or []:
        txt = str(seg.get("text", "")).strip()
        add_len = len(txt) + 8
        if current and (len(current) >= max_items or (current_chars + add_len) > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(seg)
        current_chars += add_len
    if current:
        chunks.append(current)
    return chunks


def _correct_srt_with_gemini_batched(
    api_key: str,
    prompt: str,
    file_segments: list,
    model: str = "gemini-2.5-flash",
    status_cb=None,
    progress_cb=None,
):
    if not file_segments:
        return []

    chunks = _chunk_srt_segments(file_segments)
    try:
        if progress_cb:
            progress_cb(0)
    except Exception:
        pass

    corrected_lines = []
    for idx, chunk in enumerate(chunks):
        try:
            if status_cb:
                status_cb(
                    f"Korekta Gemini SRT: paczka {idx + 1}/{len(chunks)} (segmentów: {len(chunk)})",
                    "info"
                )
        except Exception:
            pass

        numbered = []
        for i, seg in enumerate(chunk, start=1):
            numbered.append(f"{i}. {str(seg.get('text', '')).strip()}")

        batch_prompt = (
            prompt.strip()
            + "\n\nINSTRUKCJA: Otrzymasz numerowaną listę segmentów SRT."
            + " Zwróć WYŁĄCZNIE JSON-ową listę stringów, bez komentarzy i bez markdown."
            + " Każdy element listy musi odpowiadać jednemu wejściowemu segmentowi, w tej samej kolejności."
            + f"\nTo jest paczka {idx + 1}/{len(chunks)}."
        )

        response = _send_to_gemini(api_key, batch_prompt, "\n".join(numbered), model=model)
        parsed = _try_parse_json_array(response)
        if not parsed:
            parsed = _parse_jsonish_list_lines(response)
        if not parsed:
            return None

        for i, seg in enumerate(chunk):
            if i < len(parsed) and str(parsed[i]).strip():
                corrected_lines.append(str(parsed[i]).strip())
            else:
                corrected_lines.append(str(seg.get("text", "")).strip())

        try:
            if progress_cb:
                pct = int(((idx + 1) / max(1, len(chunks))) * 100)
                progress_cb(max(0, min(100, pct)))
        except Exception:
            pass

        if idx < len(chunks) - 1:
            time.sleep(4.0)

    if len(corrected_lines) < len(file_segments):
        corrected_lines.extend([str(s.get("text", "")).strip() for s in file_segments[len(corrected_lines):]])
    return corrected_lines[:len(file_segments)]


def _correct_srt_with_openrouter_batched(
    api_key: str,
    prompt: str,
    file_segments: list,
    model: str = "google/gemini-2.5-flash",
    status_cb=None,
    progress_cb=None,
):
    if not file_segments:
        return []

    chunks = _chunk_srt_segments(file_segments)
    try:
        if progress_cb:
            progress_cb(0)
    except Exception:
        pass

    corrected_lines = []
    for idx, chunk in enumerate(chunks):
        try:
            if status_cb:
                status_cb(
                    f"Korekta OpenRouter SRT: paczka {idx + 1}/{len(chunks)} (segmentów: {len(chunk)})",
                    "info"
                )
        except Exception:
            pass

        numbered = []
        for i, seg in enumerate(chunk, start=1):
            numbered.append(f"{i}. {str(seg.get('text', '')).strip()}")

        batch_prompt = (
            prompt.strip()
            + "\n\nINSTRUKCJA: Otrzymasz numerowaną listę segmentów SRT."
            + " Zwróć WYŁĄCZNIE JSON-ową listę stringów, bez komentarzy i bez markdown."
            + " Każdy element listy musi odpowiadać jednemu wejściowemu segmentowi, w tej samej kolejności."
            + f"\nTo jest paczka {idx + 1}/{len(chunks)}."
        )

        response = _send_to_openrouter(api_key, batch_prompt, "\n".join(numbered), model=model)
        parsed = _try_parse_json_array(response)
        if not parsed:
            parsed = _parse_jsonish_list_lines(response)
        if not parsed:
            return None

        for i, seg in enumerate(chunk):
            if i < len(parsed) and str(parsed[i]).strip():
                corrected_lines.append(str(parsed[i]).strip())
            else:
                corrected_lines.append(str(seg.get("text", "")).strip())

        try:
            if progress_cb:
                pct = int(((idx + 1) / max(1, len(chunks))) * 100)
                progress_cb(max(0, min(100, pct)))
        except Exception:
            pass

        if idx < len(chunks) - 1:
            time.sleep(2.0)

    if len(corrected_lines) < len(file_segments):
        corrected_lines.extend([str(s.get("text", "")).strip() for s in file_segments[len(corrected_lines):]])
    return corrected_lines[:len(file_segments)]

# Global cache for models
nllb_translator_cache = {}
helsinki_translator_cache = {}

class TranscriptionThread(QThread):
    progress_signal = Signal(int)
    status_signal = Signal(str, str)
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
                                text, segments = load_srt(local_path)
                                guessed_lang = self.config.translation_src_lang_code if self.config.translation_src_lang_code != 'auto' else self.config.src_lang_code
                                if guessed_lang == 'auto':
                                    guessed_lang = 'it'
                                info = SimpleNamespace(language=guessed_lang, paragraphs=None)
                                audio_path = None
                                self.status_signal.emit(f"Wczytano plik SRT: {os.path.basename(local_path)} (segmentów: {len(segments)})", "info")
                            except Exception as e:
                                self.status_signal.emit(f"Nie udało się wczytać pliku SRT: {e}", "error")
                                continue
                        elif local_path.lower().endswith(('.txt', '.docx', '.html', '.htm')):
                            is_text_input = True
                            try:
                                ext = os.path.splitext(local_path)[1].lower().lstrip('.')
                                if ext == 'htm':
                                    ext = 'html'
                                text, segments = _extract_saved_text(local_path, ext)
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
                                    f"Pominięto plik {os.path.basename(local_path)}: model transkrypcji ustawiony na 'Brak' obsługuje tylko wejścia tekstowe (TXT/DOCX/HTML/SRT).",
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
                    if not is_srt_input and not is_text_input:
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
                                    # Create a status callback wrapper that also emits progress_signal
                                    def _make_ollama_status_cb(base_cb, show_progress):
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
                            corr_mode = getattr(self.config, 'transcription_correction', 'Brak')
                            corr_model = getattr(self.config, 'correction_ollama_model_name', '')
                            corr_prompt = getattr(self.config, 'correction_prompt', '')
                            # basic validation: user must have selected Ollama correction and provided model + prompt length
                            if False and corr_mode and 'ollama' in corr_mode.lower() and corr_model and isinstance(corr_prompt, str) and len(corr_prompt.strip()) >= 20:
                                try:
                                    # decide which text to send: prefer formatted_text if available
                                    target_text = formatted_text if formatted_text is not None else text
                                    if target_text and target_text.strip():
                                        self.status_signal.emit(f"Uruchamiam post‑editing przez Ollama (model: {corr_model})...", "info")
                                        refiner = OllamaRefiner(corr_model, status_callback=_make_ollama_status_cb(self.status_signal.emit, True))
                                        # If we have segments, prefer a JSON-array response so mapping is robust
                                        augmented_prompt = corr_prompt
                                        if segments:
                                            augmented_prompt = (
                                                corr_prompt.strip()
                                                + "\n\nINSTRUKCJA: Zwróć poprawione segmenty w postaci JSON-owej listy stringów, np. [\"seg1\", \"seg2\", ...]. "
                                                + "Każdy element listy musi odpowiadać kolejno segmentowi we wejściu. NIE dodawaj nic poza czystym JSON-em."
                                            )
                                        refined = refiner.refine(target_text, custom_prompt=augmented_prompt)
                                        # If model returned JSON, try to parse segments
                                        try:
                                            import json as _json
                                            parsed = None
                                            # try direct JSON parse
                                            try:
                                                parsed = _json.loads(refined)
                                            except Exception:
                                                # try to find first JSON array in text
                                                import re as _re
                                                m = _re.search(r"(\[.*\])", refined, _re.DOTALL)
                                                if m:
                                                    try:
                                                        parsed = _json.loads(m.group(1))
                                                    except Exception:
                                                        parsed = None
                                            if isinstance(parsed, list) and len(parsed) >= 1 and segments:
                                                # map parsed list to segments (best-effort length match)
                                                if len(parsed) == len(segments):
                                                    for si, val in enumerate(parsed):
                                                        segments[si]['text'] = str(val).strip()
                                                else:
                                                    # if sizes differ, fill as many as possible in order
                                                    for si in range(min(len(parsed), len(segments))):
                                                        segments[si]['text'] = str(parsed[si]).strip()
                                                    # leave remaining segments as-is
                                                # replace formatted_text/text with joined parsed
                                                formatted_text = '\n\n'.join([s.get('text','') for s in segments])
                                                text = formatted_text
                                        except Exception:
                                            pass
                                        if refined and refined.strip():
                                            # Use refined text for subsequent saving (originals and translations)
                                            formatted_text = refined
                                            text = refined
                                            try:
                                                # Also attempt to map corrected text back to segments so SRT reflects corrections
                                                                            if getattr(self.config, 'enable_forced_alignment', False) and audio_path and forced_align_refined_text:
                                                                                model_name = getattr(self.config, 'forced_alignment_model', None)
                                                                                segments = forced_align_refined_text(refined, audio_path, segments, model=model_name, status_cb=self.status_signal.emit)
                                                                            else:
                                                                                # Fallback: redistribute text heuristically across segments
                                                                                if segments:
                                                                                    segments = redistribute_text_to_segments(refined, segments)
                                            except Exception:
                                                pass
                                            # Save corrected outputs (SRT + TXT) to downloads/corrected
                                            try:
                                                corrected_dir = downloads_dir
                                                base = base_name
                                                out_srt = os.path.join(corrected_dir, f"{base}_corrected.srt")
                                                out_txt = os.path.join(corrected_dir, f"{base}_corrected.txt")
                                                # If we have per-segment data, save SRT reflecting corrected text
                                                if segments:
                                                    try:
                                                        save_srt(segments, out_srt)
                                                        # also save plain TXT built from segments
                                                        txt_body = '\n\n'.join([s.get('text','') for s in segments])
                                                        save_txt(txt_body, out_txt)
                                                    except Exception as e:
                                                        self.status_signal.emit(f"Nie udało się zapisać skorygowanych plików: {e}", "warning")
                                                else:
                                                    # fallback: save entire refined text as TXT
                                                    try:
                                                        save_txt(refined, out_txt)
                                                    except Exception as e:
                                                        self.status_signal.emit(f"Nie udało się zapisać skorygowanego TXT: {e}", "warning")
                                            except Exception:
                                                pass
                                            self.status_signal.emit("Zastosowano post‑editing przez lokalny model Ollama.", "info")
                                except Exception as e:
                                    self.status_signal.emit(f"Błąd podczas post‑editingu przez Ollama: {e}", "warning")
                            # Gemini API path (send to external Gemini using stored API key)
                            elif False and corr_mode and 'gemini' in corr_mode.lower() and getattr(self.config, 'gemini_key', None) and isinstance(corr_prompt, str) and len(corr_prompt.strip()) >= 20:
                                try:
                                    target_text = formatted_text if formatted_text is not None else text
                                    if target_text and target_text.strip():
                                        self.status_signal.emit("Uruchamiam post‑editing przez Gemini (API)...", "info")
                                        augmented_prompt = corr_prompt
                                        if segments:
                                            augmented_prompt = (
                                                corr_prompt.strip()
                                                + "\n\nINSTRUKCJA: Zwróć poprawione segmenty w postaci JSON-owej listy stringów, np. [\"seg1\", \"seg2\", ...]. "
                                                + "Każdy element listy musi odpowiadać kolejno segmentowi we wejściu. NIE dodawaj nic poza czystym JSON-em."
                                            )
                                        refined = _send_to_gemini(getattr(self.config, 'gemini_key'), augmented_prompt, target_text)
                                        if refined and refined.strip():
                                            # try same JSON parsing/mapping as with Ollama
                                            try:
                                                import json as _json
                                                parsed = None
                                                try:
                                                    parsed = _json.loads(refined)
                                                except Exception:
                                                    import re as _re
                                                    m = _re.search(r"(\[.*\])", refined, _re.DOTALL)
                                                    if m:
                                                        try:
                                                            parsed = _json.loads(m.group(1))
                                                        except Exception:
                                                            parsed = None
                                                if isinstance(parsed, list) and len(parsed) >= 1 and segments:
                                                    if len(parsed) == len(segments):
                                                        for si, val in enumerate(parsed):
                                                            segments[si]['text'] = str(val).strip()
                                                    else:
                                                        for si in range(min(len(parsed), len(segments))):
                                                            segments[si]['text'] = str(parsed[si]).strip()
                                                    formatted_text = '\n\n'.join([s.get('text','') for s in segments])
                                                    text = formatted_text
                                            except Exception:
                                                pass
                                            # Use refined text for subsequent saving
                                            formatted_text = refined
                                            text = refined
                                            try:
                                                if getattr(self.config, 'enable_forced_alignment', False) and audio_path and forced_align_refined_text:
                                                    model_name = getattr(self.config, 'forced_alignment_model', None)
                                                    segments = forced_align_refined_text(refined, audio_path, segments, model=model_name, status_cb=self.status_signal.emit)
                                                else:
                                                    if segments:
                                                        segments = redistribute_text_to_segments(refined, segments)
                                            except Exception:
                                                pass
                                            # save corrected outputs
                                            try:
                                                corrected_dir = downloads_dir
                                                base = base_name
                                                out_srt = os.path.join(corrected_dir, f"{base}_corrected.srt")
                                                out_txt = os.path.join(corrected_dir, f"{base}_corrected.txt")
                                                if segments:
                                                    try:
                                                        save_srt(segments, out_srt)
                                                        txt_body = '\n\n'.join([s.get('text','') for s in segments])
                                                        save_txt(txt_body, out_txt)
                                                    except Exception as e:
                                                        self.status_signal.emit(f"Nie udało się zapisać skorygowanych plików: {e}", "warning")
                                                else:
                                                    try:
                                                        save_txt(refined, out_txt)
                                                    except Exception as e:
                                                        self.status_signal.emit(f"Nie udało się zapisać skorygowanego TXT: {e}", "warning")
                                            except Exception:
                                                pass
                                            self.status_signal.emit("Zastosowano post‑editing przez Gemini (API).", "info")
                                except Exception as e:
                                    self.status_signal.emit(f"Błąd podczas post‑editingu przez Gemini: {e}", "warning")
                        except Exception:
                            pass

                        # If paragraphing was requested for this job and paragraphs were computed,
                        # overwrite the original TXT/DOCX/HTML outputs with paragraph text instead
                        paras = getattr(info, 'paragraphs', None)
                        use_paragraphs = bool(paras) and getattr(self.config, 'enable_paragraphing', False)

                        for fmt in self.config.formats_original:
                            ext = fmt.lower()
                            path = os.path.join(downloads_dir, f"{base_name}_original.{ext}")
                            try:
                                if ext == "txt":
                                    if use_paragraphs:
                                        save_txt(paragraphs_to_plaintext(paras), path)
                                    else:
                                        save_txt(formatted_text if formatted_text is not None else text, path)
                                elif ext == "docx":
                                    if use_paragraphs:
                                        save_docx(paragraphs_to_plaintext(paras), path)
                                    else:
                                        save_docx(formatted_text if formatted_text is not None else text, path)
                                elif ext == "html":
                                    if use_paragraphs:
                                        # build simple HTML for paragraphs
                                        try:
                                            from html import escape
                                            html_parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>"]
                                            for p in paras:
                                                speaker = ''
                                                try:
                                                    sp = p.get('speakers')
                                                    if sp:
                                                        speaker = f"<strong>{escape(','.join(sorted(list(sp))))}:</strong> "
                                                except Exception:
                                                    speaker = ''
                                                html_parts.append(f"<p>{speaker}{escape(p.get('text',''))}</p>")
                                            html_parts.append("</body></html>")
                                            para_html = '\n'.join(html_parts)
                                            save_html(para_html, path)
                                        except Exception:
                                            # fallback to saving plain paragraph text
                                            save_html(paragraphs_to_plaintext(paras), path)
                                    else:
                                        save_html(formatted_text if formatted_text is not None else text, path)
                                elif ext == "srt" and segments:
                                    save_srt(segments, path)
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
                            corr_mode = getattr(self.config, 'transcription_correction', 'Brak')
                            corr_model = getattr(self.config, 'correction_ollama_model_name', '')
                            corr_prompt = getattr(self.config, 'correction_prompt', '')

                            if corr_mode and corr_mode != 'Brak':
                                if not isinstance(corr_prompt, str) or len(corr_prompt.strip()) < 20:
                                    self.status_signal.emit("Korekta pominięta: prompt jest pusty lub za krótki (min. 20 znaków).", "warning")
                                else:
                                    self.status_signal.emit("Rozpoczynam korektę na podstawie zapisanych plików transkryptu...", "info")
                                    corrected_dir = downloads_dir

                                    # Init provider once when possible
                                    ollama_refiner = None
                                    if 'ollama' in corr_mode.lower():
                                        if not corr_model:
                                            self.status_signal.emit("Korekta Ollama pominięta: brak wybranego modelu.", "warning")
                                        else:
                                            ollama_refiner = OllamaRefiner(corr_model, status_callback=_make_ollama_status_cb(self.status_signal.emit, True))

                                    correction_inputs = []
                                    # Standard path: from saved *_original.* outputs
                                    for fmt in self.config.formats_original:
                                        ext = fmt.lower()
                                        in_path = os.path.join(downloads_dir, f"{base_name}_original.{ext}")
                                        if os.path.exists(in_path):
                                            correction_inputs.append((in_path, ext))

                                    # Extra path: when transcription model is 'Brak' and user selected an existing text file,
                                    # allow correction directly from that source file (SRT/DOCX/HTML/TXT).
                                    try:
                                        if not file_or_url.startswith("http") and str(getattr(self.config, 'transcription_model', '')).strip() == 'Brak':
                                            src_ext = os.path.splitext(local_path)[1].lower().lstrip('.')
                                            if src_ext == 'htm':
                                                src_ext = 'html'
                                            if src_ext in {'srt', 'docx', 'html', 'txt'}:
                                                src_path = local_path
                                                if os.path.exists(src_path):
                                                    correction_inputs.append((src_path, src_ext))
                                    except Exception:
                                        pass

                                    # De-duplicate by absolute path keeping order
                                    seen_inputs = set()
                                    unique_inputs = []
                                    for in_path, ext in correction_inputs:
                                        key = os.path.abspath(in_path)
                                        if key in seen_inputs:
                                            continue
                                        seen_inputs.add(key)
                                        unique_inputs.append((in_path, ext))

                                    for in_path, ext in unique_inputs:

                                        try:
                                            file_text, file_segments = _extract_saved_text(in_path, ext)
                                        except Exception as e:
                                            self.status_signal.emit(f"Korekta: nie udało się odczytać pliku {os.path.basename(in_path)}: {e}", "warning")
                                            continue

                                        if not file_text or not file_text.strip():
                                            self.status_signal.emit(f"Korekta: pomijam pusty plik {os.path.basename(in_path)}.", "warning")
                                            continue

                                        refined = ""
                                        prompt_for_file = corr_prompt.strip()
                                        if ext == 'srt' and file_segments:
                                            prompt_for_file += (
                                                "\n\nINSTRUKCJA: Zwróć poprawione segmenty w postaci JSON-owej listy stringów, "
                                                "np. [\"seg1\", \"seg2\", ...]. Każdy element listy musi odpowiadać "
                                                "kolejno segmentowi wejściowemu. NIE dodawaj nic poza czystym JSON-em."
                                            )

                                        try:
                                            if 'ollama' in corr_mode.lower() and ollama_refiner is not None:
                                                self.status_signal.emit(f"Korekta ({ext.upper()}): wysyłam do Ollama...", "info")
                                                refined = ollama_refiner.refine(file_text, custom_prompt=prompt_for_file)
                                            elif 'gemini' in corr_mode.lower():
                                                gem_key = getattr(self.config, 'gemini_key', None)
                                                if not gem_key:
                                                    self.status_signal.emit("Korekta Gemini pominięta: brak klucza API.", "warning")
                                                    continue
                                                now_ts = time.time()
                                                if now_ts < gemini_rate_limited_until:
                                                    wait_left = int(max(1, gemini_rate_limited_until - now_ts))
                                                    self.status_signal.emit(f"Korekta Gemini pominięta tymczasowo (aktywny cooldown po 429: ~{wait_left}s).", "warning")
                                                    continue
                                                self.status_signal.emit(f"Korekta ({ext.upper()}): wysyłam do Gemini...", "info")
                                                if ext == 'srt' and file_segments:
                                                    parsed_list = _correct_srt_with_gemini_batched(
                                                        gem_key,
                                                        corr_prompt,
                                                        file_segments,
                                                        model="gemini-2.5-flash",
                                                        status_cb=self.status_signal.emit,
                                                        progress_cb=self.progress_signal.emit
                                                    )
                                                    if parsed_list:
                                                        refined = "\n\n".join(parsed_list)
                                                    else:
                                                        refined = ""
                                                else:
                                                    refined = _send_to_gemini(gem_key, prompt_for_file, file_text, model="gemini-2.5-flash")
                                            elif 'openrouter' in corr_mode.lower():
                                                or_key = getattr(self.config, 'openrouter_key', None)
                                                or_model = getattr(self.config, 'openrouter_model_name', None) or "google/gemini-2.5-flash"
                                                if not or_key:
                                                    self.status_signal.emit("Korekta OpenRouter pominięta: brak klucza API.", "warning")
                                                    continue
                                                self.status_signal.emit(f"Korekta ({ext.upper()}): wysyłam do OpenRouter (model: {or_model})...", "info")
                                                if ext == 'srt' and file_segments:
                                                    parsed_list = _correct_srt_with_openrouter_batched(
                                                        or_key,
                                                        corr_prompt,
                                                        file_segments,
                                                        model=or_model,
                                                        status_cb=self.status_signal.emit,
                                                        progress_cb=self.progress_signal.emit
                                                    )
                                                    if parsed_list:
                                                        refined = "\n\n".join(parsed_list)
                                                    else:
                                                        refined = ""
                                                else:
                                                    refined = _send_to_openrouter(or_key, prompt_for_file, file_text, model=or_model)
                                            else:
                                                self.status_signal.emit(f"Nieobsługiwany tryb korekty: {corr_mode}", "warning")
                                                break
                                        except httpx.HTTPStatusError as e:
                                            code = e.response.status_code if e.response is not None else None
                                            if code == 429:
                                                retry_after = None
                                                try:
                                                    hdr = e.response.headers.get("Retry-After") if e.response is not None else None
                                                    retry_after = float(hdr) if hdr else None
                                                except Exception:
                                                    retry_after = None
                                                cooldown_s = retry_after if retry_after and retry_after > 0 else 65.0
                                                cooldown_s = max(20.0, min(cooldown_s, 180.0))
                                                gemini_rate_limited_until = time.time() + cooldown_s
                                                self.status_signal.emit(
                                                    f"Korekta Gemini: przekroczony limit zapytań (429). Włączam cooldown ~{int(cooldown_s)}s.",
                                                    "warning"
                                                )
                                            else:
                                                if 'openrouter' in str(corr_mode).lower():
                                                    provider_key = getattr(self.config, 'openrouter_key', None)
                                                else:
                                                    provider_key = getattr(self.config, 'gemini_key', None)
                                                self.status_signal.emit(
                                                    f"Korekta ({ext.upper()}) nie powiodła się: {_redact_api_key_in_message(str(e), provider_key)}",
                                                    "warning"
                                                )
                                            continue
                                        except Exception as e:
                                            if 'openrouter' in str(corr_mode).lower():
                                                provider_key = getattr(self.config, 'openrouter_key', None)
                                            else:
                                                provider_key = getattr(self.config, 'gemini_key', None)
                                            self.status_signal.emit(
                                                f"Korekta ({ext.upper()}) nie powiodła się: {_redact_api_key_in_message(str(e), provider_key)}",
                                                "warning"
                                            )
                                            continue

                                        if not refined or not refined.strip():
                                            self.status_signal.emit(f"Korekta ({ext.upper()}) zwróciła pusty wynik.", "warning")
                                            continue

                                        out_path = os.path.join(corrected_dir, f"{base_name}_corrected.{ext}")
                                        try:
                                            if ext == 'txt':
                                                save_txt(refined, out_path)
                                            elif ext == 'docx':
                                                save_docx(refined, out_path)
                                            elif ext == 'html':
                                                save_html(refined, out_path)
                                            elif ext == 'srt':
                                                parsed_list = _try_parse_json_array(refined)
                                                if not parsed_list:
                                                    parsed_list = _parse_jsonish_list_lines(refined)
                                                if parsed_list and file_segments:
                                                    corr_segments = []
                                                    for i_seg, seg in enumerate(file_segments):
                                                        txt_val = str(parsed_list[i_seg]).strip() if i_seg < len(parsed_list) else seg.get('text', '')
                                                        corr_segments.append({"start": seg.get("start", 0), "end": seg.get("end", 0), "text": txt_val})
                                                    save_srt(corr_segments, out_path)
                                                    # keep runtime segments in sync for downstream translation if needed
                                                    segments = corr_segments
                                                    text = "\n\n".join([s.get('text', '') for s in corr_segments])
                                                elif file_segments:
                                                    # Last fallback: sanitize obvious JSON wrappers before redistribution
                                                    sanitized = refined
                                                    parsed_lines = _parse_jsonish_list_lines(refined)
                                                    if parsed_lines:
                                                        sanitized = "\n\n".join(parsed_lines)
                                                    corr_segments = redistribute_text_to_segments(sanitized, file_segments)
                                                    save_srt(corr_segments, out_path)
                                                    segments = corr_segments
                                                    text = "\n\n".join([s.get('text', '') for s in corr_segments])
                                                else:
                                                    # fallback if SRT parsing failed
                                                    save_txt(refined, os.path.join(corrected_dir, f"{base_name}_corrected.txt"))
                                            self.status_signal.emit(f"Zapisano korektę: {os.path.basename(out_path)}", "success")
                                        except Exception as e:
                                            self.status_signal.emit(f"Nie udało się zapisać korekty ({ext.upper()}): {e}", "warning")
                        except Exception as e:
                            self.status_signal.emit(f"Błąd krytyczny kroku korekty: {e}", "warning")

                    if self.config.translation_model != "Brak":
                        release_whisper_model()
                        if self._is_stopped: break
                        translated_text, translated_segments = translate(self.config, text, segments, info, self.status_signal, self.progress_signal, self.finished_signal, lambda: self._is_stopped)

                    if self._is_stopped: break

                    if self.config.formats_translated and (translated_text is not None or translated_segments is not None):
                        self.status_signal.emit("Zapisywanie przetłumaczonych plików...", "info")
                        if self._is_stopped: break
                        for fmt in self.config.formats_translated:
                            ext = fmt.lower()
                            path = os.path.join(downloads_dir, f"{base_name}_translation.{ext}")
                            if ext in ["txt", "docx", "html"] and translated_text is not None:
                                if ext == "txt":
                                    save_txt(translated_text, path)
                                elif ext == "docx":
                                    save_docx(translated_text, path)
                                elif ext == "html":
                                    save_html(translated_text, path)
                            elif ext == "srt" and translated_segments is not None:
                                save_srt(translated_segments, path)

                    if self._is_stopped: break

                    if self.config.summary_model != "Brak" and text:
                        summary_text = None
                        try:
                            corr_mode_now = str(getattr(self.config, 'transcription_correction', 'Brak') or '').lower()
                            if self.config.summary_model == "Gemini (API)":
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
                            self.status_signal.emit("Zapisywanie plików streszczenia...", "info")
                            for fmt in self.config.formats_summary:
                                ext = fmt.lower()
                                path = os.path.join(downloads_dir, f"{base_name}_summary.{ext}")
                                if ext == "txt":
                                    save_txt(summary_text, path)
                                elif ext == "docx":
                                    save_docx(summary_text, path)
                                elif ext == "html":
                                    save_html(summary_text, path)
                    
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

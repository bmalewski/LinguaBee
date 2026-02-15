import os
import httpx
import time
from ollama_summarizer import OllamaSummarizer
try:
    from bart_summarizer import BartSummarizer
except Exception:
    BartSummarizer = None
from config import models_dir


def _language_lock_instruction(language_name: str, language_code: str) -> str:
    return (
        "WAŻNE: Wygeneruj odpowiedź WYŁĄCZNIE w języku "
        f"{language_name} (kod: {language_code}). "
        "Nie używaj innego języka i nie mieszaj języków."
    )


def _split_text_paragraph_chunks(text: str, max_chars: int = 5000):
    raw = (text or "").strip()
    if not raw:
        return []

    paras = [p.strip() for p in raw.split("\n\n") if p and p.strip()]
    if not paras:
        paras = [raw]

    chunks = []
    cur = []
    cur_len = 0
    for p in paras:
        p_len = len(p) + 2
        if cur and (cur_len + p_len) > max_chars:
            chunks.append("\n\n".join(cur))
            cur = []
            cur_len = 0
        cur.append(p)
        cur_len += p_len
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def summarize(config, text_to_summarize, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    if config.summary_model == "Ollama (lokalny)":
        return summarize_ollama(config, text_to_summarize, whisper_info, status_signal, progress_signal, finished_signal, is_stopped)

    if config.summary_model == "Gemini (API)":
        return summarize_gemini(config, text_to_summarize, whisper_info, status_signal, progress_signal, finished_signal, is_stopped)

    if config.summary_model == "OpenRouter (API)":
        return summarize_openrouter(config, text_to_summarize, whisper_info, status_signal, progress_signal, finished_signal, is_stopped)

    if config.summary_model == "BART (lokalny)":
        return summarize_bart(config, text_to_summarize, whisper_info, status_signal, progress_signal, finished_signal, is_stopped)

    return None


def _send_to_gemini_summary(api_key: str, prompt: str, input_text: str, model: str = "gemini-2.5-flash") -> str:
    if not api_key:
        return ""
    normalized_model = model.strip() if isinstance(model, str) and model.strip() else "gemini-2.5-flash"
    if normalized_model.startswith("models/"):
        normalized_model = normalized_model.split("/", 1)[1]

    payload = {
        "contents": [{"parts": [{"text": prompt.strip() + "\n\n" + input_text.strip()}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
    }

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{normalized_model}:generateContent"
    last_error = None
    with httpx.Client(timeout=120) as c:
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
                txt = []

                def _collect_text(node):
                    if isinstance(node, dict):
                        t = node.get("text")
                        if isinstance(t, str) and t.strip():
                            txt.append(t.strip())
                        for v in node.values():
                            _collect_text(v)
                    elif isinstance(node, list):
                        for it in node:
                            _collect_text(it)

                _collect_text(j.get("candidates") if isinstance(j, dict) else j)
                return "\n".join(txt).strip()
            except Exception as e:
                last_error = e

    if last_error is not None:
        raise last_error
    return ""


def _send_to_openrouter_summary(api_key: str, prompt: str, input_text: str, model: str = "google/gemini-2.5-flash") -> str:
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
    with httpx.Client(timeout=120) as c:
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


def summarize_openrouter(config, text_to_summarize, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    api_key = getattr(config, 'openrouter_key', None)
    if not api_key:
        status_signal.emit("Błąd: Brak klucza API OpenRouter dla streszczenia.", "error")
        return None

    custom_prompt = getattr(config, 'summary_openrouter_prompt', None)
    if not custom_prompt or not isinstance(custom_prompt, str) or not custom_prompt.strip():
        status_signal.emit("Błąd: Brak promptu OpenRouter dla sekcji streszczenie.", "error")
        return None

    model_name = getattr(config, 'summary_openrouter_model_name', None) or "google/gemini-2.5-flash"
    lang_map = {"en": "angielski", "pl": "polski", "de": "niemiecki", "fr": "francuski", "es": "hiszpański", "it": "włoski", "uk": "ukraiński", "ru": "rosyjski", "ja": "japoński", "ko": "koreański", "la": "łaciński"}
    language_name = lang_map.get(getattr(config, 'summary_lang_code', 'pl'), getattr(config, 'summary_lang_code', 'pl'))
    language_code = getattr(config, 'summary_lang_code', 'pl')

    prompt = custom_prompt.strip().replace("{language_name}", language_name).replace("{language_code}", language_code)
    prompt = prompt.replace("{text}", "")
    prompt = _language_lock_instruction(language_name, language_code) + "\n\n" + prompt
    input_text = text_to_summarize

    try:
        status_signal.emit(f"Tworzenie streszczenia (OpenRouter API, model: {model_name})...", "info")
        progress_signal.emit(0)
        out = _send_to_openrouter_summary(api_key, prompt, input_text, model=model_name)
        if not out or not out.strip():
            status_signal.emit("Nie udało się wygenerować streszczenia przez OpenRouter (pusta odpowiedź).", "warning")
            return None
        progress_signal.emit(100)
        return out.strip()
    except Exception as e:
        msg = str(e)
        try:
            if api_key:
                msg = msg.replace(api_key, "REDACTED")
        except Exception:
            pass
        status_signal.emit(f"Błąd OpenRouter (streszczenie): {msg}", "error")
        return None


def summarize_gemini(config, text_to_summarize, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    api_key = getattr(config, 'gemini_key', None)
    if not api_key:
        status_signal.emit("Błąd: Brak klucza API Gemini dla streszczenia.", "error")
        return None

    custom_prompt = getattr(config, 'summary_gemini_prompt', None)
    if not custom_prompt or not isinstance(custom_prompt, str) or not custom_prompt.strip():
        status_signal.emit("Błąd: Brak promptu Gemini dla sekcji streszczenie.", "error")
        return None

    lang_map = {"en": "angielski", "pl": "polski", "de": "niemiecki", "fr": "francuski", "es": "hiszpański", "it": "włoski", "uk": "ukraiński", "ru": "rosyjski", "ja": "japoński", "ko": "koreański", "la": "łaciński"}
    language_name = lang_map.get(getattr(config, 'summary_lang_code', 'pl'), getattr(config, 'summary_lang_code', 'pl'))
    language_code = getattr(config, 'summary_lang_code', 'pl')

    prompt = custom_prompt.strip().replace("{language_name}", language_name).replace("{language_code}", language_code)
    # Zawsze wysyłamy pełny transkrypt jako osobny input.
    # Jeśli użytkownik zostawił placeholder {text} w promptcie, usuwamy go,
    # aby nie dublować treści i nie wydłużać niepotrzebnie promptu.
    prompt = prompt.replace("{text}", "")
    prompt = _language_lock_instruction(language_name, language_code) + "\n\n" + prompt
    input_text = text_to_summarize

    try:
        status_signal.emit("Tworzenie streszczenia (Gemini API)...", "info")
        progress_signal.emit(0)

        out = _send_to_gemini_summary(api_key, prompt, input_text)
        if not out or not out.strip():
            status_signal.emit("Nie udało się wygenerować streszczenia przez Gemini (pusta odpowiedź).", "warning")
            return None
        progress_signal.emit(100)
        return out.strip()
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code if e.response is not None else None
        if status_code == 429:
            status_signal.emit("Błąd Gemini (streszczenie): przekroczony limit zapytań (429). Spróbuj ponownie za chwilę.", "warning")
            return None
        msg = str(e)
        try:
            if api_key:
                msg = msg.replace(api_key, "REDACTED")
        except Exception:
            pass
        status_signal.emit(f"Błąd Gemini (streszczenie): {msg}", "error")
        return None
    except Exception as e:
        msg = str(e)
        try:
            if api_key:
                msg = msg.replace(api_key, "REDACTED")
        except Exception:
            pass
        status_signal.emit(f"Błąd Gemini (streszczenie): {msg}", "error")
        return None


def summarize_bart(config, text_to_summarize, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    """Use local BART model for summarization."""
    if BartSummarizer is None:
        status_signal.emit("Błąd: brakuje modułu bart_summarizer. Upewnij się, że plik istnieje.", "error")
        return None

    model_name = config.hf_summary_model_name

    summarizer = None
    try:
        summarizer = BartSummarizer(
            model_name=model_name,
            device=config.hf_summary_device,
            device_index=config.hf_summary_device_index,
            max_length=config.hf_summary_max_length,
            min_length=config.hf_summary_min_length,
            num_beams=config.hf_summary_num_beams,
            status_callback=status_signal.emit
        )
        
        status_signal.emit(f"Tworzenie streszczenia (model: {model_name})...", "info")
        progress_signal.emit(0) # Progress is not granular, so we simulate 0 -> 100

        summary_text = summarizer.summarize(
            text_to_summarize,
            summary_lang_code=config.summary_lang_code,
            custom_prompt=getattr(config, 'bart_summary_prompt', None)
        )

        progress_signal.emit(100)
        return summary_text
    except Exception as e:
        status_signal.emit(f"Wystąpił krytyczny błąd podczas streszczenia: {e}", "error")
        return None
    finally:
        if summarizer:
            summarizer.release()
            del summarizer
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

def _get_lang_name_by_code(lang_code):
    lang_map = {"en": "angielski", "pl": "polski", "de": "niemiecki", "fr": "francuski", "es": "hiszpański", "it": "włoski", "uk": "ukraiński", "ru": "rosyjski", "ja": "japoński", "ko": "koreański", "la": "łaciński"}
    return lang_map.get(lang_code, lang_code)

def summarize_ollama(config, text_to_summarize, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    if not config.ollama_summary_model_name:
        status_signal.emit("Błąd: Nazwa modelu Ollama dla streszczenia nie została podana.", "error")
        return None

    custom_prompt = getattr(config, 'ollama_summary_prompt', None)
    if not custom_prompt or not isinstance(custom_prompt, str) or not custom_prompt.strip():
        status_signal.emit("Błąd: Brak promptu dla streszczenia Ollama. Uzupełnij prompt w oknie ustawień/szablonu.", "error")
        return None
    status_signal.emit(f"Używam promptu Ollama z ustawień (długość: {len(custom_prompt.strip())} znaków).", "info")
    
    summary_lang_full = _get_lang_name_by_code(config.summary_lang_code)
    
    summarizer = OllamaSummarizer(config.ollama_summary_model_name, status_callback=status_signal.emit)
    
    try:
        status_signal.emit(f"Tworzenie streszczenia (Ollama, model: {config.ollama_summary_model_name})...", "info")
        progress_signal.emit(0)
        
        summary_data = summarizer.summarize(
            text_to_summarize,
            summary_lang_full,
            custom_prompt=custom_prompt
        )

        if not summary_data:
            status_signal.emit("Nie udało się wygenerować streszczenia (pusta odpowiedź).", "warning")
            return None

        # Preferred path: plain text from user-defined prompt.
        if isinstance(summary_data, str):
            out = summary_data.strip()
            if not out:
                status_signal.emit("Nie udało się wygenerować streszczenia (pusta odpowiedź tekstowa).", "warning")
                return None
            progress_signal.emit(100)
            return out

        # Legacy compatibility: old JSON structure with 'propozycje'.
        if not isinstance(summary_data, dict) or "propozycje" not in summary_data:
            status_signal.emit("Otrzymano nieobsługiwany format odpowiedzi streszczenia — zapisuję odpowiedź jako tekst.", "warning")
            out = str(summary_data).strip()
            progress_signal.emit(100)
            return out if out else None

        output_string = ""
        propozycje = summary_data.get("propozycje", {})

        kategorie = {
            "formalne": "Formalne / Oficjalne",
            "intrygujace": "Intrygujące / Zaczepne",
            "zabawne": "Zabawne / Ciekawe"
        }

        for klucz, nazwa_kategorii in kategorie.items():
            kategoria_data = propozycje.get(klucz)
            if kategoria_data and isinstance(kategoria_data, dict):
                tytul = kategoria_data.get("tytul")
                opis = kategoria_data.get("opis")

                if tytul or opis:
                    output_string += f"### {nazwa_kategorii} ###\n\n"
                    if tytul:
                        output_string += "--- Tytuł ---"
                        output_string += f"{tytul}\n\n"
                    
                    if opis:
                        output_string += "--- Opis ---"
                        output_string += f"{opis}\n\n"

        progress_signal.emit(100)
        return output_string.strip()

    except Exception as e:
        status_signal.emit(f"Nieoczekiwany błąd w summarize_ollama: {repr(e)}", "error")
        return None

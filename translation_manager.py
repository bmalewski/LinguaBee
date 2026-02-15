from text_utils import chunk_text, add_missing_spaces
import httpx
import time
from nllb_translator import NLLBTranslator
from ollama_translator import OllamaTranslator
try:
    from helsinki_translator import HelsinkiTranslator
except ImportError:
    HelsinkiTranslator = None

# This cache will be managed by the worker thread
extern_nllb_translator_cache = {}
extern_helsinki_translator_cache = {}

def translate(config, original_text, original_segments, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    if config.translation_model == "NLLB (lokalny)":
        return translate_nllb(config, original_text, original_segments, whisper_info, status_signal, progress_signal, finished_signal, is_stopped)
    elif config.translation_model == "Ollama (lokalny)":
        return translate_ollama(config, original_text, original_segments, whisper_info, status_signal, progress_signal, finished_signal, is_stopped)
    elif config.translation_model == "Helsinki (lokalny)":
        return translate_helsinki(config, original_text, original_segments, whisper_info, status_signal, progress_signal, finished_signal, is_stopped)
    elif config.translation_model == "OpenRouter (API)":
        return translate_openrouter(config, original_text, original_segments, whisper_info, status_signal, progress_signal, finished_signal, is_stopped)
    return None, None


def _send_to_openrouter_translate(api_key: str, model: str, src_lang_full: str, tgt_lang_full: str, input_text: str) -> str:
    if not api_key:
        return ""

    normalized_model = model.strip() if isinstance(model, str) and model.strip() else "google/gemini-2.5-flash"
    system_prompt = (
        "Jesteś tłumaczem. Tłumacz wiernie i naturalnie. "
        "Zwróć wyłącznie przetłumaczony tekst bez komentarzy i bez dodatkowych wyjaśnień."
    )
    user_prompt = (
        f"Przetłumacz z języka {src_lang_full} na język {tgt_lang_full}.\n\n"
        f"Tekst:\n{input_text.strip()}"
    )

    payload = {
        "model": normalized_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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


def _normalize_translated_segments(original_segments, translated_segments_for_srt):
    """Ensure translated segments align with original_segments order and timestamps.

    - If translated_segments_for_srt is empty or None, create empty entries for each original segment.
    - If lengths match, map by index (preserve ordering).
    - Otherwise try to match by (start,end) keys; if a match isn't found, fill with empty text.
    """
    if not original_segments:
        return translated_segments_for_srt

    if not translated_segments_for_srt:
        return [{"start": seg.get("start"), "end": seg.get("end"), "text": ""} for seg in original_segments]

    # If same length, assume ordering corresponds and align by index
    if len(translated_segments_for_srt) == len(original_segments):
        normalized = []
        for orig, trans in zip(original_segments, translated_segments_for_srt):
            normalized.append({"start": orig.get("start"), "end": orig.get("end"), "text": trans.get("text", "")})
        return normalized

    # Otherwise try to match by timestamps
    lookup = {}
    for t in translated_segments_for_srt:
        key = (t.get("start"), t.get("end"))
        lookup.setdefault(key, []).append(t.get("text", ""))

    normalized = []
    for seg in original_segments:
        key = (seg.get("start"), seg.get("end"))
        texts = lookup.get(key)
        if texts and len(texts) > 0:
            text = texts.pop(0)
        else:
            text = ""
        normalized.append({"start": seg.get("start"), "end": seg.get("end"), "text": text})
    return normalized

def translate_nllb(config, original_text, original_segments, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    global extern_nllb_translator_cache
    nllb_model_mapping = {
        "distilled-600M": {"ct2_id": "entai2965/nllb-200-distilled-600M-ctranslate2", "tokenizer_id": "facebook/nllb-200-distilled-600M"},
        "1.3B": {"ct2_id": "entai2965/nllb-200-distilled-1.3B-ctranslate2", "tokenizer_id": "facebook/nllb-200-1.3B"},
        "3.3B": {"ct2_id": "entai2965/nllb-200-3.3B-ctranslate2", "tokenizer_id": "facebook/nllb-200-3.3B"}
    }
    selected_variant_info = nllb_model_mapping.get(config.nllb_variant)
    if not selected_variant_info:
        finished_signal.emit(f"Nieznany wariant NLLB: {config.nllb_variant}", "error")
        return None, None

    ct2_model_id, hf_tokenizer_id = selected_variant_info["ct2_id"], selected_variant_info["tokenizer_id"]
    translator_key = f"{ct2_model_id}_{config.nllb_device}_{config.nllb_device_index}"

    if translator_key not in extern_nllb_translator_cache:
        try:
            compute_type = "float16" if config.nllb_device == "cuda" else "float32"
            extern_nllb_translator_cache[translator_key] = NLLBTranslator(
                ct2_model_id, 
                hf_tokenizer_id, 
                device=config.nllb_device, 
                device_index=config.nllb_device_index, 
                compute_type=compute_type, 
                status_callback=status_signal.emit
            )
        except Exception as e:
            finished_signal.emit(f"Nie udało się załadować modelu NLLB: {e}", "error")
            return None, None
    
    translator = extern_nllb_translator_cache[translator_key]

    lang_code_map = {"en": "eng_Latn", "pl": "pol_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "it": "ita_Latn", "uk": "ukr_Cyrl", "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang", "la": "lat_Latn"}
    t_src_lang = config.translation_src_lang_code
    if t_src_lang and t_src_lang != 'auto':
        src_lang_code = t_src_lang
    else:
        src_lang_code = config.src_lang_code
        if src_lang_code == 'auto':
            src_lang_code = whisper_info.language
    
    nllb_src_lang = lang_code_map.get(src_lang_code)
    nllb_tgt_lang = lang_code_map.get(config.tgt_lang_code)

    if not nllb_src_lang or not nllb_tgt_lang:
        finished_signal.emit(f"Nieobsługiwany kod języka dla NLLB. Kod źródłowy: '{src_lang_code}', Kod docelowy: '{config.tgt_lang_code}'", "error")
        return None, None

    translated_text_full, translated_segments_for_srt = None, None
    formats_lower = [f.lower() for f in config.formats_translated]
    did_translation = False

    if any(f in formats_lower for f in ["txt", "docx", "html"]):
        did_translation = True
        status_signal.emit("Tłumaczenie tekstu (NLLB)...", "info")
        progress_signal.emit(0)

        paras = getattr(whisper_info, 'paragraphs', None)

        if paras:
            # If paragraphs with speaker info are available, translate paragraph by paragraph
            status_signal.emit("Tłumaczenie z zachowaniem struktury akapitów i mówców...", "info")
            
            original_para_texts = [p.get('text', '').strip() for p in paras]
            translated_para_texts = []
            batch_size = 4 if config.nllb_variant == "3.3B" and config.nllb_device == "cuda" else 8
            total_paras = len(original_para_texts)

            for i in range(0, total_paras, batch_size):
                if is_stopped(): return None, None
                
                batch_texts = original_para_texts[i:i+batch_size]
                
                # We need to handle empty texts within the batch carefully
                non_empty_batch = [text for text in batch_texts if text]
                translated_batch = []
                if non_empty_batch:
                    translated_batch = translator.translate_batch(non_empty_batch, nllb_src_lang, nllb_tgt_lang)
                
                # Re-align translated texts with original (potentially empty) paragraphs
                it = iter(translated_batch)
                for text in batch_texts:
                    translated_para_texts.append(next(it) if text else "")

                progress = ((i + len(batch_texts)) / total_paras) * 100
                progress_signal.emit(int(progress))

            if is_stopped(): return None, None

            # Reconstruct paragraphs with translated text and speaker info
            translated_paragraphs = []
            for i, p in enumerate(paras):
                speaker_label = ""
                speakers = p.get('speakers')
                if speakers:
                    speaker_label = f"{','.join(sorted(list(speakers)))}: "
                
                translated_text = translated_para_texts[i] if i < len(translated_para_texts) else ""
                translated_paragraphs.append(f"{speaker_label}{add_missing_spaces(translated_text)}")

            translated_text_full = "\n\n".join(translated_paragraphs)
            progress_signal.emit(100)

        else:
            # Fallback to translating the whole text if no paragraphs are available
            status_signal.emit("Brak informacji o akapitach, tłumaczę cały tekst...", "info")
            chunks = chunk_text(original_text)
            translated_chunks = []
            total_chunks = len(chunks)
            batch_size = 4 if config.nllb_variant == "3.3B" and config.nllb_device == "cuda" else 8
            for i in range(0, total_chunks, batch_size):
                if is_stopped(): return None, None
                batch_chunks = chunks[i:i+batch_size]
                translated_chunks.extend(translator.translate_batch(batch_chunks, nllb_src_lang, nllb_tgt_lang))
                if total_chunks > 0:
                    progress = ((i + len(batch_chunks)) / total_chunks) * 100
                    progress_signal.emit(int(progress))
            if is_stopped(): return None, None
            translated_text_full = add_missing_spaces("".join(translated_chunks))
            progress_signal.emit(100)

    if "srt" in formats_lower and original_segments:
        did_translation = True
        status_signal.emit("Tłumaczenie segmentów dla SRT (NLLB)...", "info")
        progress_signal.emit(0)
        translated_segments_for_srt = []
        
        batch_size = 4 if config.nllb_variant == "3.3B" and config.nllb_device == "cuda" else 8
        total_segments = len(original_segments)
        
        for i in range(0, total_segments, batch_size):
            if is_stopped(): return None, None
            
            batch = original_segments[i:i+batch_size]
            texts_to_translate = [seg.get("text", "").strip() for seg in batch]
            
            translated_texts = translator.translate_batch(texts_to_translate, nllb_src_lang, nllb_tgt_lang)

            if len(translated_texts) == len(batch):
                for seg, translated_text in zip(batch, translated_texts):
                    translated_segments_for_srt.append({
                        "start": seg["start"], 
                        "end": seg["end"], 
                        "text": add_missing_spaces(translated_text)
                    })
            else:
                status_signal.emit(f"Błąd: Niedopasowanie liczby segmentów w batchu {i//batch_size + 1} dla NLLB. Spodziewano się {len(batch)}, otrzymano {len(translated_texts)}. Przełączam na tryb pojedynczy dla tego batcha.", "warning")
                for seg in batch:
                    if is_stopped(): return None, None
                    segment_text = seg.get("text", "").strip()
                    translated_text = add_missing_spaces(translator.translate(segment_text, nllb_src_lang, nllb_tgt_lang)) if segment_text else ""
                    translated_segments_for_srt.append({"start": seg["start"], "end": seg["end"], "text": translated_text})

            progress = min(int(((i + batch_size) / total_segments) * 100), 100)
            progress_signal.emit(progress)

        progress_signal.emit(100)

    # Fallback: if no per-format translation was requested but a translation model was selected,
    # users typically expect the text to be translated. Perform a full-text translation as a fallback.
    if not did_translation and original_text and original_text.strip():
        # Only do this fallback if the worker actually requested translation (translation_model selected)
        status_signal.emit("Tłumaczenie tekstu (NLLB) — fallback pełnego tłumaczenia...", "info")
        progress_signal.emit(0)
        chunks = chunk_text(original_text)
        translated_chunks = []
        total_chunks = len(chunks)
        batch_size = 4 if config.nllb_variant == "3.3B" and config.nllb_device == "cuda" else 8
        for i in range(0, total_chunks, batch_size):
            if is_stopped(): return None, None
            batch_chunks = chunks[i:i+batch_size]
            translated_chunks.extend(translator.translate_batch(batch_chunks, nllb_src_lang, nllb_tgt_lang))
            if total_chunks > 0:
                progress = ((i + len(batch_chunks)) / total_chunks) * 100
                progress_signal.emit(int(progress))
        if is_stopped(): return None, None
        translated_text_full = add_missing_spaces("".join(translated_chunks))
        progress_signal.emit(100)

    status_signal.emit("Zwalnianie pamięci VRAM po tłumaczeniu NLLB...", "info")
    translator.release()
    extern_nllb_translator_cache.clear()
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # Ensure alignment with original segments
    return translated_text_full, _normalize_translated_segments(original_segments, translated_segments_for_srt)

def translate_ollama(config, original_text, original_segments, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    if not config.ollama_model_name:
        status_signal.emit("Błąd: Nazwa modelu Ollama nie została podana.", "error")
        return None, None

    # Determine language names for the prompt
    src_lang_code = config.src_lang_code
    if src_lang_code == 'auto':
        src_lang_code = whisper_info.language
    
    lang_map = {"en": "angielski", "pl": "polski", "de": "niemiecki", "fr": "francuski", "es": "hiszpański", "it": "włoski", "uk": "ukraiński", "ru": "rosyjski", "ja": "japoński", "ko": "koreański", "la": "łaciński"}
    src_lang_full = lang_map.get(src_lang_code, src_lang_code)
    tgt_lang_full = lang_map.get(config.tgt_lang_code, config.tgt_lang_code)
    
    translated_text_full, translated_segments_for_srt = None, None
    ollama_translator = OllamaTranslator(config.ollama_model_name)
    formats_lower = [f.lower() for f in config.formats_translated]

    try:
        if any(f in formats_lower for f in ["txt", "docx", "html"]):
            status_signal.emit(f"Tłumaczenie tekstu (Ollama, model: {config.ollama_model_name})...", "info")
            progress_signal.emit(0)

            paras = getattr(whisper_info, 'paragraphs', None)

            if paras:
                status_signal.emit("Tłumaczenie z zachowaniem struktury akapitów i mówców...", "info")
                translated_paragraphs = []
                total_paras = len(paras)
                for i, p in enumerate(paras):
                    if is_stopped(): return None, None
                    
                    original_para_text = p.get('text', '').strip()
                    if not original_para_text:
                        translated_paragraphs.append("")
                        continue
                    
                    translated_para_text = ollama_translator.translate(original_para_text, src_lang_full, tgt_lang_full)

                    speaker_label = ""
                    speakers = p.get('speakers')
                    if speakers:
                        speaker_label = f"{','.join(sorted(list(speakers)))}: "

                    translated_paragraphs.append(f"{speaker_label}{add_missing_spaces(translated_para_text)}")

                    if total_paras > 0:
                        progress = ((i + 1) / total_paras) * 100
                        progress_signal.emit(int(progress))

                if is_stopped(): return None, None
                translated_text_full = "\n\n".join(translated_paragraphs)
                progress_signal.emit(100)
            else:
                status_signal.emit("Brak informacji o akapitach, tłumaczę cały tekst...", "info")
                chunks = chunk_text(original_text, max_chunk_size=1000)
                translated_chunks = []
                total_chunks = len(chunks)
                for i, chk in enumerate(chunks):
                    if is_stopped(): return None, None
                    translated_chunks.append(ollama_translator.translate(chk, src_lang_full, tgt_lang_full))
                    if total_chunks > 0:
                        progress = ((i + 1) / total_chunks) * 100
                        progress_signal.emit(int(progress))
                translated_text_full = add_missing_spaces(" ".join(translated_chunks))
                progress_signal.emit(100)

        if "srt" in formats_lower and original_segments:
            status_signal.emit(f"Tłumaczenie segmentów dla SRT (Ollama, model: {config.ollama_model_name})...", "info")
            progress_signal.emit(0)
            translated_segments_for_srt = []
            
            batch_size = 1
            total_segments = len(original_segments)
            
            for i in range(0, total_segments, batch_size):
                if is_stopped(): return None, None
                
                batch = original_segments[i:i+batch_size]
                texts_to_translate = [seg.get("text", "").strip() for seg in batch]
                
                translated_texts = ollama_translator.translate_batch(texts_to_translate, src_lang_full, tgt_lang_full)

                if len(translated_texts) == len(batch):
                    for seg, translated_text in zip(batch, translated_texts):
                        translated_segments_for_srt.append({
                            "start": seg["start"], 
                            "end": seg["end"], 
                            "text": add_missing_spaces(translated_text)
                        })
                else:
                    status_signal.emit(f"Błąd: Niedopasowanie liczby segmentów w batchu {i//batch_size + 1} dla Ollama. Spodziewano się {len(batch)}, otrzymano {len(translated_texts)}. Przełączam na tryb pojedynczy dla tego batcha.", "warning")
                    for seg in batch:
                        if is_stopped(): return None, None
                        segment_text = seg.get("text", "").strip()
                        translated_text = ""
                        if segment_text:
                            translated_text = add_missing_spaces(ollama_translator.translate(segment_text, src_lang_full, tgt_lang_full))
                        translated_segments_for_srt.append({"start": seg["start"], "end": seg["end"], "text": translated_text})

                progress = min(int(((i + batch_size) / total_segments) * 100), 100)
                progress_signal.emit(progress)

            progress_signal.emit(100)

    except Exception as e:
        status_signal.emit(f"Nieoczekiwany błąd w translate_ollama: {repr(e)}", "error")
        return None, None
        
    return translated_text_full, _normalize_translated_segments(original_segments, translated_segments_for_srt)


def translate_openrouter(config, original_text, original_segments, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    api_key = getattr(config, 'openrouter_key', None)
    if not api_key:
        status_signal.emit("Błąd: Brak klucza API OpenRouter dla tłumaczenia.", "error")
        return None, None

    model_name = getattr(config, 'translation_openrouter_model_name', None) or "google/gemini-2.5-flash"

    src_lang_code = config.translation_src_lang_code
    if not src_lang_code or src_lang_code == 'auto':
        src_lang_code = config.src_lang_code
        if src_lang_code == 'auto':
            src_lang_code = whisper_info.language

    lang_map = {
        "en": "angielski", "pl": "polski", "de": "niemiecki", "fr": "francuski", "es": "hiszpański",
        "it": "włoski", "uk": "ukraiński", "ru": "rosyjski", "ja": "japoński", "ko": "koreański",
        "la": "łaciński", "zh": "chiński", "pt": "portugalski", "tr": "turecki"
    }
    src_lang_full = lang_map.get(src_lang_code, src_lang_code)
    tgt_lang_full = lang_map.get(config.tgt_lang_code, config.tgt_lang_code)

    translated_text_full, translated_segments_for_srt = None, None
    formats_lower = [f.lower() for f in config.formats_translated]

    try:
        if any(f in formats_lower for f in ["txt", "docx", "html"]):
            status_signal.emit(f"Tłumaczenie tekstu (OpenRouter, model: {model_name})...", "info")
            progress_signal.emit(0)

            paras = getattr(whisper_info, 'paragraphs', None)
            if paras:
                translated_paragraphs = []
                total_paras = len(paras)
                for i, p in enumerate(paras):
                    if is_stopped():
                        return None, None

                    original_para_text = p.get('text', '').strip()
                    if not original_para_text:
                        translated_paragraphs.append("")
                        continue

                    translated_para_text = _send_to_openrouter_translate(
                        api_key,
                        model_name,
                        src_lang_full,
                        tgt_lang_full,
                        original_para_text,
                    )

                    speaker_label = ""
                    speakers = p.get('speakers')
                    if speakers:
                        speaker_label = f"{','.join(sorted(list(speakers)))}: "

                    translated_paragraphs.append(f"{speaker_label}{add_missing_spaces(translated_para_text)}")
                    if total_paras > 0:
                        progress_signal.emit(int(((i + 1) / total_paras) * 100))

                translated_text_full = "\n\n".join(translated_paragraphs)
                progress_signal.emit(100)
            else:
                chunks = chunk_text(original_text, max_chunk_size=1000)
                translated_chunks = []
                total_chunks = len(chunks)
                for i, chk in enumerate(chunks):
                    if is_stopped():
                        return None, None
                    translated_chunks.append(
                        _send_to_openrouter_translate(api_key, model_name, src_lang_full, tgt_lang_full, chk)
                    )
                    if total_chunks > 0:
                        progress_signal.emit(int(((i + 1) / total_chunks) * 100))
                translated_text_full = add_missing_spaces(" ".join(translated_chunks))
                progress_signal.emit(100)

        if "srt" in formats_lower and original_segments:
            status_signal.emit(f"Tłumaczenie segmentów dla SRT (OpenRouter, model: {model_name})...", "info")
            progress_signal.emit(0)
            translated_segments_for_srt = []
            total_segments = len(original_segments)

            for i, seg in enumerate(original_segments):
                if is_stopped():
                    return None, None
                src_text = seg.get("text", "").strip()
                out_text = ""
                if src_text:
                    out_text = _send_to_openrouter_translate(api_key, model_name, src_lang_full, tgt_lang_full, src_text)
                    out_text = add_missing_spaces(out_text)
                translated_segments_for_srt.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": out_text,
                })
                if total_segments > 0:
                    progress_signal.emit(int(((i + 1) / total_segments) * 100))

            progress_signal.emit(100)

    except Exception as e:
        msg = str(e)
        try:
            if api_key:
                msg = msg.replace(api_key, "REDACTED")
        except Exception:
            pass
        status_signal.emit(f"Nieoczekiwany błąd w translate_openrouter: {msg}", "error")
        return None, None

    return translated_text_full, _normalize_translated_segments(original_segments, translated_segments_for_srt)

def translate_helsinki(config, original_text, original_segments, whisper_info, status_signal, progress_signal, finished_signal, is_stopped):
    global extern_helsinki_translator_cache
    if HelsinkiTranslator is None:
        finished_signal.emit("Błąd: Nie można zaimportować modułu HelsinkiTranslator.", "error")
        return None, None

    # Określenie języka źródłowego
    src_lang_code = config.translation_src_lang_code
    if not src_lang_code or src_lang_code == 'auto':
        src_lang_code = whisper_info.language

    # Mapowanie na modele Helsinki-NLP dla tłumaczenia na polski
    helsinki_model_mapping = {
        "en": "Helsinki-NLP/opus-mt-en-pl",
        "de": "Helsinki-NLP/opus-mt-de-pl",
        "fr": "Helsinki-NLP/opus-mt-fr-pl",
        "es": "Helsinki-NLP/opus-mt-es-pl",
        "it": "Helsinki-NLP/opus-mt-it-pl",
    }

    if config.tgt_lang_code != "pl":
        finished_signal.emit(f"Model Helsinki-NLP w tej aplikacji obsługuje tylko tłumaczenie na język polski (pl). Wybrano: {config.tgt_lang_code}", "error")
        return None, None

    model_name = helsinki_model_mapping.get(src_lang_code)
    if not model_name:
        finished_signal.emit(f"Model Helsinki-NLP nie obsługuje tłumaczenia z '{src_lang_code}' na polski w tej aplikacji.", "error")
        return None, None

    # Używamy tego samego urządzenia co dla NLLB jako domyślne
    device = config.nllb_device
    device_index = config.nllb_device_index
    translator_key = f"{model_name}_{device}_{device_index}"

    if translator_key not in extern_helsinki_translator_cache:
        try:
            translator_device = f"{device}:{device_index}" if device == "cuda" else device
            extern_helsinki_translator_cache[translator_key] = HelsinkiTranslator(
                model_name=model_name,
                device=translator_device,
                status_callback=status_signal.emit
            )
        except Exception as e:
            finished_signal.emit(f"Nie udało się załadować modelu Helsinki-NLP: {e}", "error")
            return None, None
    
    translator = extern_helsinki_translator_cache[translator_key]

    translated_text_full, translated_segments_for_srt = None, None
    formats_lower = [f.lower() for f in config.formats_translated]
    did_translation = False

    if any(f in formats_lower for f in ["txt", "docx", "html"]):
        did_translation = True
        status_signal.emit(f"Tłumaczenie tekstu (Helsinki-NLP, model: {model_name})...", "info")
        progress_signal.emit(0)
        
        paras = getattr(whisper_info, 'paragraphs', None)

        if paras:
            status_signal.emit("Tłumaczenie z zachowaniem struktury akapitów i mówców...", "info")
            
            original_para_texts = [p.get('text', '').strip() for p in paras]
            
            translated_para_texts = []
            batch_size = 8
            total_paras = len(original_para_texts)

            for i in range(0, total_paras, batch_size):
                if is_stopped(): return None, None
                batch_texts = original_para_texts[i:i+batch_size]
                
                non_empty_batch = [t for t in batch_texts if t]
                translated_batch = translator.translate_batch(non_empty_batch) if non_empty_batch else []
                
                # Re-align translated texts with original (potentially empty) paragraphs
                it = iter(translated_batch)
                for original_text in batch_texts:
                    translated_para_texts.append(next(it) if original_text else "")

                progress = ((i + len(batch_texts)) / total_paras) * 100
                progress_signal.emit(int(progress))

            if is_stopped(): return None, None
            
            translated_paragraphs_formatted = []
            for i, p in enumerate(paras):
                speaker_label = ""
                speakers = p.get('speakers')
                if speakers:
                    speaker_label = f"{','.join(sorted(list(speakers)))}: "
                
                translated_text = translated_para_texts[i] if i < len(translated_para_texts) else ""
                translated_paragraphs_formatted.append(f"{speaker_label}{add_missing_spaces(translated_text)}")
            
            translated_text_full = "\n\n".join(translated_paragraphs_formatted)
            progress_signal.emit(100)

        else:
            status_signal.emit("Brak informacji o akapitach, tłumaczę cały tekst...", "info")
            chunks = chunk_text(original_text)
            translated_chunks = []
            total_chunks = len(chunks)
            for i in range(0, total_chunks, 8): # Przetwarzanie w batchach po 8
                if is_stopped(): return None, None
                batch_chunks = chunks[i:i+8]
                translated_batch = translator.translate_batch(batch_chunks)
                translated_chunks.extend(translated_batch)
                if total_chunks > 0:
                    progress = ((i + len(batch_chunks)) / total_chunks) * 100
                    progress_signal.emit(int(progress))
            if is_stopped(): return None, None
            translated_text_full = add_missing_spaces(" ".join(translated_chunks))
            progress_signal.emit(100)

    if "srt" in formats_lower and original_segments:
        did_translation = True
        status_signal.emit(f"Tłumaczenie segmentów dla SRT (Helsinki-NLP, model: {model_name})...", "info")
        progress_signal.emit(0)
        
        batch_size = 16 # Modele Helsinki są małe, możemy użyć większego batcha
        texts_to_translate = [seg.get("text", "").strip() for seg in original_segments]
        translated_texts = []
        total_segments = len(texts_to_translate)

        for i in range(0, total_segments, batch_size):
            if is_stopped(): return None, None
            batch_texts = texts_to_translate[i:i+batch_size]
            translated_batch = translator.translate_batch(batch_texts)
            translated_texts.extend(translated_batch)
            progress = min(int(((i + len(batch_texts)) / total_segments) * 100), 100)
            progress_signal.emit(progress)

        translated_segments_for_srt = []
        if len(original_segments) == len(translated_texts):
            for seg, translated_text in zip(original_segments, translated_texts):
                translated_segments_for_srt.append({
                    "start": seg["start"], 
                    "end": seg["end"], 
                    "text": add_missing_spaces(translated_text)
                })
        progress_signal.emit(100)

    return translated_text_full, _normalize_translated_segments(original_segments, translated_segments_for_srt)

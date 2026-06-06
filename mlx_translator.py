"""MLX-based translation backend for Apple Silicon (M1/M2/M3/M4).

Uses mlx-lm to run quantised LLM models (default: Gemma 3 12B 4-bit)
directly on the Neural Engine / GPU cores of Apple Silicon chips without
any round-trip through Ollama or a remote API.

Install dependency:
    pip install mlx-lm

Supported models (HuggingFace model IDs):
    mlx-community/gemma-3-12b-it-4bit   (default, ~7 GB unified memory)
    mlx-community/gemma-3-4b-it-4bit    (~2.5 GB, faster, lower quality)
    mlx-community/mistral-7b-instruct-v0.3-4bit
    mlx-community/Llama-3.2-3B-Instruct-4bit
"""
from __future__ import annotations

import re
import time

DEFAULT_MLX_MODEL = "mlx-community/gemma-3-12b-it-4bit"
_MAX_NEW_TOKENS = 2048

# Module-level cache so the model is loaded only once per process.
_mlx_model_cache: dict = {}


def _get_model_and_tokenizer(model_id: str):
    """Load and cache model + tokenizer."""
    if model_id not in _mlx_model_cache:
        try:
            from mlx_lm import load
        except ImportError as e:
            raise ImportError(
                "mlx-lm nie jest zainstalowane. Uruchom: pip install mlx-lm"
            ) from e
        model, tokenizer = load(model_id)
        _mlx_model_cache[model_id] = (model, tokenizer)
    return _mlx_model_cache[model_id]


def _generate(model_id: str, prompt: str, max_tokens: int = _MAX_NEW_TOKENS) -> str:
    from mlx_lm import generate
    model, tokenizer = _get_model_and_tokenizer(model_id)
    result = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return (result or "").strip()


def _lang_map(code: str) -> str:
    m = {
        "pl": "polski", "en": "angielski", "de": "niemiecki", "fr": "francuski",
        "es": "hiszpański", "it": "włoski", "uk": "ukraiński", "ru": "rosyjski",
        "ja": "japoński", "ko": "koreański", "la": "łaciński", "zh": "chiński",
        "pt": "portugalski", "tr": "turecki",
    }
    return m.get(code, code)


def translate_mlx(
    config,
    original_text: str,
    original_segments: list,
    whisper_info,
    status_signal,
    progress_signal,
    finished_signal,
    is_stopped,
):
    """Translate using a local MLX model on Apple Silicon.

    Mirrors the signature of translate_nllb / translate_ollama so it can be
    dispatched from translation_manager.translate().
    """
    model_id = getattr(config, "mlx_model_id", None) or DEFAULT_MLX_MODEL

    src_code = getattr(config, "translation_src_lang_code", None) or getattr(config, "src_lang_code", "auto")
    if src_code == "auto":
        src_code = getattr(whisper_info, "language", None) or "en"
    tgt_code = getattr(config, "tgt_lang_code", "pl")

    src_lang = _lang_map(src_code)
    tgt_lang = _lang_map(tgt_code)

    formats_lower = [f.lower() for f in (config.formats_translated or [])]
    translated_text_full = None
    translated_segments_for_srt = None

    try:
        # Check if the model is already cached locally to warn the user before
        # a potentially large download starts (the tqdm progress bar only shows
        # in the terminal, not in the app UI).
        _model_is_cached = False
        try:
            from huggingface_hub import snapshot_download as _snap
            _snap(model_id, local_files_only=True)
            _model_is_cached = True
        except Exception:
            pass

        if _model_is_cached:
            status_signal.emit(f"Ładowanie modelu MLX: {model_id} ...", "info")
        else:
            status_signal.emit(
                f"⚠️ Model MLX '{model_id}' nie jest zapisany lokalnie. "
                "Trwa pobieranie z HuggingFace (może to potrwać kilka minut – "
                "model waży kilka GB). Postęp pobierania widoczny w terminalu.",
                "warning",
            )

        _get_model_and_tokenizer(model_id)  # warm-up / load
        status_signal.emit("Model MLX załadowany.", "info")
    except Exception as e:
        finished_signal.emit(f"Błąd ładowania modelu MLX: {e}", "error")
        return None, None

    # --- TXT / DOCX translation ---
    if any(f in formats_lower for f in ["txt", "docx"]):
        if not original_text or not original_text.strip():
            finished_signal.emit("Brak tekstu do tłumaczenia.", "error")
            return None, None

        status_signal.emit(f"Tłumaczenie tekstu (MLX, model: {model_id})...", "info")
        progress_signal.emit(0)

        # Split into paragraphs to stay within context window
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", original_text.strip()) if p.strip()]
        translated_parts = []
        total = len(paragraphs)
        start_time = time.time()

        for idx, para in enumerate(paragraphs):
            if is_stopped():
                finished_signal.emit("Tłumaczenie zatrzymane.", "warning")
                return None, None

            prompt = (
                f"Przetłumacz poniższy tekst z języka {src_lang} na język {tgt_lang}. "
                "Zwróć wyłącznie tłumaczenie bez komentarzy.\n\nTekst:\n" + para
            )
            try:
                out = _generate(model_id, prompt)
                translated_parts.append(out)
            except Exception as e:
                status_signal.emit(f"Błąd MLX dla akapitu {idx + 1}: {e}", "warning")
                translated_parts.append(para)  # fallback: keep original

            pct = int(((idx + 1) / total) * 100)
            progress_signal.emit(pct)
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 0.001)
            eta = (total - idx - 1) / rate if rate > 0 else None
            eta_str = f"{int(eta // 60):02}:{int(eta % 60):02}" if eta else "--:--"
            status_signal.emit(f"MLX tłumaczenie: {idx + 1}/{total} akapitów | ETA: {eta_str}", "info")

        translated_text_full = "\n\n".join(translated_parts)

    # --- SRT translation ---
    if "srt" in formats_lower and original_segments:
        status_signal.emit(f"Tłumaczenie napisów SRT (MLX, model: {model_id})...", "info")
        progress_signal.emit(0)

        translated_segments_for_srt = []
        total = len(original_segments)
        start_time = time.time()
        BATCH = 20  # translate N segments at once to reduce round-trips

        for batch_start in range(0, total, BATCH):
            if is_stopped():
                finished_signal.emit("Tłumaczenie zatrzymane.", "warning")
                return translated_text_full, None

            batch = original_segments[batch_start: batch_start + BATCH]
            lines = "\n".join(f"{i + 1}. {seg.get('text', '').strip()}" for i, seg in enumerate(batch))
            prompt = (
                f"Przetłumacz kolejne napisy z języka {src_lang} na język {tgt_lang}. "
                "Każda linia to osobna kwestia. Zachowaj numerację. "
                "Zwróć wyłącznie ponumerowane tłumaczenia, bez dodatkowych komentarzy.\n\n"
                + lines
            )
            try:
                raw = _generate(model_id, prompt)
                # Parse "N. translated text" lines
                parsed: dict[int, str] = {}
                for line in raw.splitlines():
                    m = re.match(r"^(\d+)\.\s*(.*)", line.strip())
                    if m:
                        parsed[int(m.group(1))] = m.group(2).strip()
                for i, seg in enumerate(batch):
                    text = parsed.get(i + 1, seg.get("text", ""))
                    translated_segments_for_srt.append({
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": text,
                    })
            except Exception as e:
                status_signal.emit(f"Błąd MLX dla segmentów {batch_start}–{batch_start + BATCH}: {e}", "warning")
                for seg in batch:
                    translated_segments_for_srt.append({
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", ""),
                    })

            done = min(batch_start + BATCH, total)
            pct = int((done / total) * 100)
            progress_signal.emit(pct)
            elapsed = time.time() - start_time
            rate = done / max(elapsed, 0.001)
            eta = (total - done) / rate if rate > 0 else None
            eta_str = f"{int(eta // 60):02}:{int(eta % 60):02}" if eta else "--:--"
            status_signal.emit(f"MLX SRT: {done}/{total} segmentów | ETA: {eta_str}", "info")

    progress_signal.emit(100)
    return translated_text_full, translated_segments_for_srt


def release_mlx_model(model_id: str | None = None):
    """Remove a cached MLX model to free unified memory."""
    global _mlx_model_cache
    if model_id:
        _mlx_model_cache.pop(model_id, None)
    else:
        _mlx_model_cache.clear()

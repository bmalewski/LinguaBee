"""llama-cpp-python translation backend for NVIDIA GPU (CUDA) and CPU.

Uses llama-cpp-python to run GGUF-quantised models locally.
With n_gpu_layers=-1 all layers are offloaded to the GPU — ideal for
an RTX 3090 Ti (24 GB VRAM).

Install dependency:
    # CPU only:
    pip install llama-cpp-python
    # CUDA (compile with GPU support):
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

Default model (downloaded automatically on first run via HuggingFace):
    bartowski/gemma-3-12b-it-GGUF  Q4_K_M  (~7.5 GB)

Other good options for Polish:
    bartowski/Llama-3.1-8B-Instruct-GGUF       Q4_K_M  (8 GB VRAM)
    bartowski/TowerInstruct-7B-v0.2-GGUF        Q4_K_M  (best translation quality)
    bartowski/mistral-7b-instruct-v0.3-GGUF     Q4_K_M
"""
from __future__ import annotations

import os
import re
import time

DEFAULT_LLAMA_MODEL_REPO = "bartowski/gemma-3-12b-it-GGUF"
DEFAULT_LLAMA_MODEL_FILE = "gemma-3-12b-it-Q4_K_M.gguf"
_MAX_TOKENS = 2048
_CONTEXT_SIZE = 8192

# Module-level cache: model_path → Llama instance
_llama_cache: dict = {}


def _resolve_model_path(model_repo: str, model_file: str) -> str:
    """Return local path to GGUF file, downloading from HuggingFace if needed."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=model_repo, filename=model_file)
        return path
    except ImportError as e:
        raise ImportError(
            "huggingface_hub nie jest zainstalowane. Uruchom: pip install huggingface-hub"
        ) from e


def _get_llama(model_path: str, n_gpu_layers: int = -1, n_ctx: int = _CONTEXT_SIZE):
    """Load and cache a Llama model instance."""
    cache_key = f"{model_path}:{n_gpu_layers}"
    if cache_key not in _llama_cache:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python nie jest zainstalowane.\n"
                "CPU: pip install llama-cpp-python\n"
                "CUDA: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall"
            ) from e
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        _llama_cache[cache_key] = llm
    return _llama_cache[cache_key]


def _generate(llm, prompt: str, max_tokens: int = _MAX_TOKENS) -> str:
    output = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["<end_of_turn>", "</s>", "<|im_end|>"],
        echo=False,
    )
    return (output["choices"][0]["text"] or "").strip()


def _lang_map(code: str) -> str:
    m = {
        "pl": "polski", "en": "angielski", "de": "niemiecki", "fr": "francuski",
        "es": "hiszpański", "it": "włoski", "uk": "ukraiński", "ru": "rosyjski",
        "ja": "japoński", "ko": "koreański", "la": "łaciński", "zh": "chiński",
        "pt": "portugalski", "tr": "turecki",
    }
    return m.get(code, code)


def translate_llama_cpp(
    config,
    original_text: str,
    original_segments: list,
    whisper_info,
    status_signal,
    progress_signal,
    finished_signal,
    is_stopped,
):
    """Translate using a local GGUF model via llama-cpp-python.

    Mirrors the signature of translate_nllb / translate_mlx.
    """
    model_repo = getattr(config, "llama_cpp_model_repo", None) or DEFAULT_LLAMA_MODEL_REPO
    model_file = getattr(config, "llama_cpp_model_file", None) or DEFAULT_LLAMA_MODEL_FILE
    n_gpu_layers = int(getattr(config, "llama_cpp_n_gpu_layers", -1))

    src_code = getattr(config, "translation_src_lang_code", None) or getattr(config, "src_lang_code", "auto")
    if src_code == "auto":
        src_code = getattr(whisper_info, "language", None) or "en"
    tgt_code = getattr(config, "tgt_lang_code", "pl")

    src_lang = _lang_map(src_code)
    tgt_lang = _lang_map(tgt_code)

    formats_lower = [f.lower() for f in (config.formats_translated or [])]
    translated_text_full = None
    translated_segments_for_srt = None

    # --- Load model ---
    try:
        status_signal.emit(f"Rozwiązuję ścieżkę modelu GGUF: {model_repo} / {model_file} ...", "info")
        model_path = _resolve_model_path(model_repo, model_file)
        status_signal.emit(f"Ładowanie modelu llama.cpp: {os.path.basename(model_path)} (n_gpu_layers={n_gpu_layers})...", "info")
        llm = _get_llama(model_path, n_gpu_layers=n_gpu_layers)
        status_signal.emit("Model llama.cpp załadowany.", "info")
    except Exception as e:
        finished_signal.emit(f"Błąd ładowania modelu llama.cpp: {e}", "error")
        return None, None

    # --- TXT / DOCX ---
    if any(f in formats_lower for f in ["txt", "docx"]):
        if not original_text or not original_text.strip():
            finished_signal.emit("Brak tekstu do tłumaczenia.", "error")
            return None, None

        status_signal.emit(f"Tłumaczenie tekstu (llama.cpp, model: {model_file})...", "info")
        progress_signal.emit(0)

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", original_text.strip()) if p.strip()]
        translated_parts = []
        total = len(paragraphs)
        start_time = time.time()

        for idx, para in enumerate(paragraphs):
            if is_stopped():
                finished_signal.emit("Tłumaczenie zatrzymane.", "warning")
                return None, None

            prompt = (
                f"Translate the following text from {src_lang} to {tgt_lang}. "
                "Return only the translation without any commentary.\n\nText:\n" + para
            )
            try:
                out = _generate(llm, prompt)
                translated_parts.append(out)
            except Exception as e:
                status_signal.emit(f"Błąd llama.cpp dla akapitu {idx + 1}: {e}", "warning")
                translated_parts.append(para)

            pct = int(((idx + 1) / total) * 100)
            progress_signal.emit(pct)
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 0.001)
            eta = (total - idx - 1) / rate if rate > 0 else None
            eta_str = f"{int(eta // 60):02}:{int(eta % 60):02}" if eta else "--:--"
            status_signal.emit(f"llama.cpp tłumaczenie: {idx + 1}/{total} | ETA: {eta_str}", "info")

        translated_text_full = "\n\n".join(translated_parts)

    # --- SRT ---
    if "srt" in formats_lower and original_segments:
        status_signal.emit(f"Tłumaczenie napisów SRT (llama.cpp)...", "info")
        progress_signal.emit(0)

        translated_segments_for_srt = []
        total = len(original_segments)
        start_time = time.time()
        BATCH = 20

        for batch_start in range(0, total, BATCH):
            if is_stopped():
                finished_signal.emit("Tłumaczenie zatrzymane.", "warning")
                return translated_text_full, None

            batch = original_segments[batch_start: batch_start + BATCH]
            lines = "\n".join(f"{i + 1}. {seg.get('text', '').strip()}" for i, seg in enumerate(batch))
            prompt = (
                f"Translate the following subtitles from {src_lang} to {tgt_lang}. "
                "Keep the numbering. Return only numbered translations.\n\n" + lines
            )
            try:
                raw = _generate(llm, prompt)
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
                status_signal.emit(f"Błąd llama.cpp dla segmentów {batch_start}–{batch_start + BATCH}: {e}", "warning")
                for seg in batch:
                    translated_segments_for_srt.append({
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", ""),
                    })

            done = min(batch_start + BATCH, total)
            progress_signal.emit(int((done / total) * 100))
            elapsed = time.time() - start_time
            rate = done / max(elapsed, 0.001)
            eta = (total - done) / rate if rate > 0 else None
            eta_str = f"{int(eta // 60):02}:{int(eta % 60):02}" if eta else "--:--"
            status_signal.emit(f"llama.cpp SRT: {done}/{total} | ETA: {eta_str}", "info")

    progress_signal.emit(100)
    return translated_text_full, translated_segments_for_srt


def release_llama_model(model_path: str | None = None):
    """Remove cached llama.cpp instance(s) to free VRAM / RAM."""
    global _llama_cache
    if model_path:
        keys = [k for k in _llama_cache if k.startswith(model_path)]
        for k in keys:
            del _llama_cache[k]
    else:
        _llama_cache.clear()

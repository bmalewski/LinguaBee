"""
Wrapper for BART summarization model (Hugging Face).
- Downloads/caches automatically via transformers.
- Supports custom prompt with {text} placeholder.
- Uses chunked summarization for longer transcripts.
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from text_utils import chunk_text

logger = logging.getLogger(__name__)


class BartSummarizer:
    def __init__(self, model_name: str, device: str = "cpu", device_index: int = 0, max_length: int = 150, min_length: int = 30, num_beams: int = 4, status_callback=None):
        self.device_name = device
        self.device_index = device_index
        if self.device_name == "cuda" and torch.cuda.is_available():
            self.device = f"cuda:{self.device_index}"
        else:
            self.device = "cpu"

        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams

        self.tokenizer = None
        self.model = None
        self.status_callback = status_callback

        if self.status_callback:
            self.status_callback(f"Wybrano urządzenie: {self.device.upper()}", "info")

    def _ensure_loaded(self):
        if self.model is None or self.tokenizer is None:
            if self.status_callback:
                self.status_callback(f"Ładowanie modelu '{self.model_name}'. Może to potrwać chwilę przy pierwszym uruchomieniu...", "info")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            if self.status_callback:
                self.status_callback("Model załadowany pomyślnie.", "info")

    def release(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.status_callback:
            self.status_callback("Zwolniono model streszczający z pamięci.", "info")

    def _build_prompt(self, text: str, summary_lang_code: str, custom_prompt: str = None) -> str:
        lang_map = {
            "en": "angielskim", "pl": "polskim", "de": "niemieckim", "fr": "francuskim",
            "es": "hiszpańskim", "it": "włoskim", "uk": "ukraińskim"
        }
        lang_name = lang_map.get(summary_lang_code, "polskim")

        if custom_prompt and isinstance(custom_prompt, str) and custom_prompt.strip():
            base = custom_prompt.strip()
            if "{text}" in base:
                return base.replace("{text}", text)
            return base + "\n\nTekst do streszczenia:\n" + text

        return (
            f"Stwórz zwięzłe i spójne streszczenie poniższego tekstu w języku {lang_name}. "
            "Zwróć tylko gotowe streszczenie, bez komentarzy.\n\n"
            f"Tekst:\n{text}"
        )

    def summarize(self, text: str, **kwargs) -> str:
        if not text or not text.strip():
            return ""

        self._ensure_loaded()
        summary_lang = kwargs.get('summary_lang_code', 'pl')
        custom_prompt = kwargs.get('custom_prompt')

        try:
            # Chunk by characters to avoid hard truncation for long transcripts.
            chunks = chunk_text(text, max_chunk_size=1600)
            if not chunks:
                chunks = [text]

            partial_summaries = []
            total = len(chunks)
            for i, chk in enumerate(chunks, start=1):
                prompt_text = self._build_prompt(chk, summary_lang, custom_prompt)
                inputs = self.tokenizer(prompt_text, return_tensors="pt", max_length=2048, truncation=True).to(self.device)
                summary_ids = self.model.generate(
                    inputs['input_ids'],
                    num_beams=self.num_beams,
                    max_length=self.max_length,
                    min_length=min(self.min_length, max(0, self.max_length - 5)),
                    early_stopping=True
                )
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
                if summary:
                    partial_summaries.append(summary)
                if self.status_callback and total > 1:
                    self.status_callback(f"Streszczenie BART: fragment {i}/{total} zakończony.", "info")

            if not partial_summaries:
                return ""
            if len(partial_summaries) == 1:
                return partial_summaries[0]

            # Final compression pass over partial summaries.
            combined = "\n\n".join(partial_summaries)
            final_prompt = self._build_prompt(combined, summary_lang, custom_prompt)
            inputs = self.tokenizer(final_prompt, return_tensors="pt", max_length=2048, truncation=True).to(self.device)
            final_ids = self.model.generate(
                inputs['input_ids'],
                num_beams=self.num_beams,
                max_length=self.max_length,
                min_length=min(self.min_length, max(0, self.max_length - 5)),
                early_stopping=True
            )
            return self.tokenizer.decode(final_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.exception("Błąd podczas generowania streszczenia BART.")
            if self.status_callback:
                self.status_callback(f"Błąd generowania streszczenia BART: {e}", "error")
            return ""

"""
Wrapper for Helsinki-NLP translation models.
- Automatically downloads and caches models from the Hugging Face Hub.
- Uses the transformers library for inference.
- Provides a translate(text, ...) function.
- Supports device selection: 'cuda' or 'cpu'.
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

class HelsinkiTranslator:
    def __init__(self, model_name: str, device: str = "cuda", status_callback=None):
        """
        Initializes the translator with a model name from Hugging Face Hub.

        model_name: Name of the model on Hugging Face Hub (e.g., 'Helsinki-NLP/opus-mt-en-pl').
        device: 'cuda' or 'cpu'.
        status_callback: A function to emit status updates.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.status_callback = status_callback

        if self.status_callback:
            self.status_callback(f"Wybrano urządzenie dla Helsinki-NLP: {self.device.upper()}", "info")

    def _ensure_loaded(self):
        """
        Ensures that the model and tokenizer are loaded into memory.
        """
        if self.model is None or self.tokenizer is None:
            if self.status_callback:
                self.status_callback(f"Ładowanie modelu '{self.model_name}'. Może to potrwać chwilę...", "info")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
                if self.status_callback:
                    self.status_callback("Model Helsinki-NLP załadowany pomyślnie.", "info")
            except Exception as e:
                logger.exception(f"Nie udało się załadować modelu '{self.model_name}'.")
                if self.status_callback:
                    self.status_callback(f"Błąd podczas ładowania modelu Helsinki-NLP: {e}", "error")
                raise

    def release(self):
        """Releases the model and tokenizer from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.status_callback:
            self.status_callback(f"Zwolniono model Helsinki-NLP '{self.model_name}' z pamięci.", "info")
        logger.info(f"Zwolniono model Helsinki-NLP '{self.model_name}' z pamięci.")

    def translate_batch(self, texts: list, **kwargs) -> list:
        """
        Translates a batch of texts.
        """
        if not texts:
            return []

        self._ensure_loaded()
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        translated_tokens = self.model.generate(**inputs, **kwargs)
        translated_texts = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        
        return translated_texts
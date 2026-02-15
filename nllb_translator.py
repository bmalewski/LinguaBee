import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ctranslate2
from huggingface_hub import snapshot_download
from config import models_dir

# Global cache for NLLB model
nllb_model_cache = {}

class NLLBTranslator:
    def __init__(self, ct2_model_id, hf_tokenizer_id, device="cpu", device_index=0, compute_type="float32", status_callback=None):
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.ct2_model_id = ct2_model_id
        self.ct2_model_path = os.path.join(models_dir, ct2_model_id.replace("/", "_"))
        self.hf_tokenizer_id = hf_tokenizer_id
        self.status_callback = status_callback
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        global nllb_model_cache
        model_key = f"{self.ct2_model_id}_{self.device}_{self.device_index}_{self.compute_type}"

        if model_key in nllb_model_cache:
            if self.status_callback:
                self.status_callback(f"Using cached NLLB model '{self.ct2_model_id}' on {self.device}:{self.device_index}", "info")
            self.model, self.tokenizer = nllb_model_cache[model_key]
        else:
            if self.status_callback:
                self.status_callback(f"Downloading and loading NLLB model '{self.ct2_model_id}' to '{self.ct2_model_path}' on {self.device}:{self.device_index}. This may take a while...", "info")
            
            try:
                snapshot_download(repo_id=self.ct2_model_id, allow_patterns=["*.bin", "*.json"], local_dir_use_symlinks=False, local_dir=self.ct2_model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_id, cache_dir=models_dir)
                self.model = ctranslate2.Translator(self.ct2_model_path, device=self.device, device_index=self.device_index, compute_type=self.compute_type)
                
                nllb_model_cache[model_key] = (self.model, self.tokenizer)
                if self.status_callback:
                    self.status_callback("NLLB model loaded successfully using CTranslate2.", "info")
            except Exception as e:
                # Provide a clearer, actionable message when the ctranslate2-ready repo is not available
                guidance = (
                    "Failed to download or load the CTranslate2 model.\n"
                    "If you selected the 12B variant, a public ctranslate2-converted artifact may not exist under the assumed repo id.\n"
                    "Options:\n"
                    "  1) Provide a ctranslate2-ready repo on the Hugging Face hub and set the ct2_model_id accordingly.\n"
                    "  2) Convert the Hugging Face Transformers 12B weights to CTranslate2 following the CTranslate2 docs: https://opennmt.net/CTranslate2/ and upload the result to a repo.\n"
                    "  3) Use a smaller model variant that is available (e.g. 3.3B or 1.3B).\n"
                    "Error details: "
                )
                if self.status_callback:
                    self.status_callback(guidance + str(e), "error")
                raise

    def translate(self, text, src_lang_code, tgt_lang_code):
        if not self.model or not self.tokenizer:
            raise Exception("NLLB model is not loaded.")
        
        try:
            # Conservative input cap to avoid positional-encoding overflow on long segments.
            max_src_tokens = 900
            encoded_text_ids = self.tokenizer.encode(text, add_special_tokens=False)[:max_src_tokens]
            src_tokens_for_ctranslate2 = [src_lang_code] + self.tokenizer.convert_ids_to_tokens(encoded_text_ids)

            target_prefix_for_ctranslate2 = [[tgt_lang_code]]

            results = self.model.translate_batch(
                [src_tokens_for_ctranslate2],
                target_prefix=target_prefix_for_ctranslate2,
                max_batch_size=1,
                max_input_length=max_src_tokens + 1,
                max_decoding_length=1024,
                beam_size=4
            )
            
            translated_text = self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(results[0].hypotheses[0]),
                skip_special_tokens=True
            )
            return translated_text
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error during NLLB translation with CTranslate2: {e}", "error")
            return text

    def translate_batch(self, texts: list, src_lang_code, tgt_lang_code):
        if not self.model or not self.tokenizer:
            raise Exception("NLLB model is not loaded.")
        
        if not texts:
            return []

        # Progressive fallback settings: reduce memory pressure and sequence lengths.
        attempts = [
            {"max_batch_size": min(8, len(texts)), "max_src_tokens": 900, "max_decoding_length": 1024, "beam_size": 4},
            {"max_batch_size": min(4, len(texts)), "max_src_tokens": 768, "max_decoding_length": 768, "beam_size": 4},
            {"max_batch_size": 1, "max_src_tokens": 512, "max_decoding_length": 512, "beam_size": 2},
        ]

        last_error = None
        for cfg in attempts:
            try:
                encoded_texts_ids = [self.tokenizer.encode(text, add_special_tokens=False)[:cfg["max_src_tokens"]] for text in texts]
                src_tokens_for_ctranslate2 = [[src_lang_code] + self.tokenizer.convert_ids_to_tokens(ids) for ids in encoded_texts_ids]
                target_prefix_for_ctranslate2 = [[tgt_lang_code]] * len(texts)

                results = self.model.translate_batch(
                    src_tokens_for_ctranslate2,
                    target_prefix=target_prefix_for_ctranslate2,
                    max_batch_size=cfg["max_batch_size"],
                    max_input_length=cfg["max_src_tokens"] + 1,
                    max_decoding_length=cfg["max_decoding_length"],
                    beam_size=cfg["beam_size"]
                )

                translated_texts = []
                for result in results:
                    translated_text = self.tokenizer.decode(
                        self.tokenizer.convert_tokens_to_ids(result.hypotheses[0]),
                        skip_special_tokens=True
                    )
                    translated_texts.append(translated_text)
                return translated_texts
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if self.status_callback:
                    self.status_callback(
                        f"Error during NLLB batch translation with CTranslate2 (retry with smaller settings): {e}",
                        "warning"
                    )
                # Try to free VRAM between retries
                try:
                    if "out of memory" in msg and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

        # Last fallback: split input into single-item calls to isolate problematic segments.
        if len(texts) > 1:
            out = []
            for t in texts:
                out.append(self.translate(t, src_lang_code, tgt_lang_code))
            return out

        if self.status_callback and last_error is not None:
            self.status_callback(f"Error during NLLB batch translation with CTranslate2: {last_error}", "error")
        return texts

    def release(self):
        """Releases the CTranslate2 model from memory."""
        global nllb_model_cache
        model_key = f"{self.ct2_model_id}_{self.device}_{self.device_index}_{self.compute_type}"

        if self.model is not None:
            if self.status_callback:
                self.status_callback(f"Releasing NLLB model '{self.ct2_model_id}'.", "info")
            
            if model_key in nllb_model_cache:
                del nllb_model_cache[model_key]

            del self.model
            self.model = None
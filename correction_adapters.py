import re
import time
import httpx
import os
from config import downloads_dir

from ollama_refiner import OllamaRefiner
from model_response_parser import parse_list_response


class CorrectionAdapters:
    def __init__(
        self,
        config,
        status_cb,
        progress_cb,
        make_ollama_status_cb,
    ):
        self.config = config
        self.status_cb = status_cb
        self.progress_cb = progress_cb
        self.make_ollama_status_cb = make_ollama_status_cb
        self._ollama_refiner = None

    def _send_to_gemini(self, api_key: str, prompt: str, input_text: str, model: str = "gemini-2.5-flash") -> str:
        from api_client import call_gemini
        prompt_text = prompt.strip() + "\n\nTekst do poprawy:\n" + input_text.strip()
        return call_gemini(api_key, model, prompt_text, timeout=90)


    def _send_to_openrouter(self, api_key: str, prompt: str, input_text: str, model: str = "google/gemini-3.5-flash") -> str:
        from api_client import call_openrouter
        messages = [
            {"role": "system", "content": prompt.strip()},
            {"role": "user", "content": input_text.strip()},
        ]
        return call_openrouter(api_key, model, messages, timeout=90)


    def ensure_provider_ready(self, corr_mode: str) -> bool:
        mode = str(corr_mode or "").lower()
        if "ollama" not in mode:
            return True

        if self._ollama_refiner is not None:
            return True

        corr_model = getattr(self.config, "correction_ollama_model_name", "")
        if not corr_model:
            self.status_cb("Korekta Ollama pominięta: brak wybranego modelu.", "warning")
            return False

        self._ollama_refiner = OllamaRefiner(
            corr_model,
            status_callback=self.make_ollama_status_cb(self.status_cb, True),
        )
        return True

    def correct_file(self, corr_mode: str, ext: str, file_text: str, prompt_for_file: str, file_segments, gemini_rate_limited_until: float):
        mode = str(corr_mode or "").lower()
        refined = ""

        try:
            if "ollama" in mode:
                self.status_cb(f"Korekta ({ext.upper()}): wysyłam do Ollama...", "info")
                if ext in {"txt", "docx"}:
                    refined = self._correct_text_in_batched_chunks(
                        file_text,
                        prompt_for_file,
                        "Ollama",
                        lambda chunk_text, chunk_prompt: self._ollama_refiner.refine(chunk_text, custom_prompt=chunk_prompt),
                        batch_size=200,
                    )
                else:
                    refined = self._ollama_refiner.refine(file_text, custom_prompt=prompt_for_file)

            elif "gemini" in mode:
                gem_key = getattr(self.config, "gemini_key", None)
                if not gem_key:
                    self.status_cb("Korekta Gemini pominięta: brak klucza API.", "warning")
                    return "", gemini_rate_limited_until

                now_ts = time.time()
                if now_ts < gemini_rate_limited_until:
                    wait_left = int(max(1, gemini_rate_limited_until - now_ts))
                    self.status_cb(f"Korekta Gemini pominięta tymczasowo (aktywny cooldown po 429: ~{wait_left}s).", "warning")
                    return "", gemini_rate_limited_until

                self.status_cb(f"Korekta ({ext.upper()}): wysyłam do Gemini...", "info")
                if ext == "srt" and file_segments:
                    parsed_list = self._correct_srt_with_batched_provider(
                        "Gemini",
                        getattr(self.config, "correction_prompt", ""),
                        file_segments,
                        send_batch=lambda batch_prompt, numbered_text: self._send_to_gemini(
                            gem_key,
                            batch_prompt,
                            numbered_text,
                            model=getattr(self.config, 'gemini_correction_model', 'gemini-2.5-flash'),
                        ),
                        batch_size=getattr(self.config, "transcription_segment_batch_size", 200),
                        inter_batch_sleep_s=4.0,
                    )
                    refined = "\n\n".join(parsed_list) if parsed_list else ""
                elif ext in {"txt", "docx"}:
                    refined = self._correct_text_in_batched_chunks(
                        file_text,
                        prompt_for_file,
                        "Gemini",
                        lambda chunk_text, chunk_prompt: self._send_to_gemini(
                            gem_key,
                            chunk_prompt,
                            chunk_text,
                            model=getattr(self.config, 'gemini_correction_model', 'gemini-2.5-flash'),
                        ),
                        batch_size=200,
                    )
                else:
                    refined = self._send_to_gemini(gem_key, prompt_for_file, file_text, model=getattr(self.config, 'gemini_correction_model', 'gemini-2.5-flash'))

            elif "openrouter" in mode:
                or_key = getattr(self.config, "openrouter_key", None)
                or_model = getattr(self.config, "openrouter_model_name", None) or "google/gemini-3.5-flash"
                if not or_key:
                    self.status_cb("Korekta OpenRouter pominięta: brak klucza API.", "warning")
                    return "", gemini_rate_limited_until

                self.status_cb(f"Korekta ({ext.upper()}): wysyłam do OpenRouter (model: {or_model})...", "info")
                if ext == "srt" and file_segments:
                    parsed_list = self._correct_srt_with_batched_provider(
                        "OpenRouter",
                        getattr(self.config, "correction_prompt", ""),
                        file_segments,
                        send_batch=lambda batch_prompt, numbered_text: self._send_to_openrouter(
                            or_key,
                            batch_prompt,
                            numbered_text,
                            model=or_model,
                        ),
                        batch_size=getattr(self.config, "transcription_segment_batch_size", 200),
                        inter_batch_sleep_s=2.0,
                    )
                    refined = "\n\n".join(parsed_list) if parsed_list else ""
                elif ext in {"txt", "docx"}:
                    refined = self._correct_text_in_batched_chunks(
                        file_text,
                        prompt_for_file,
                        "OpenRouter",
                        lambda chunk_text, chunk_prompt: self._send_to_openrouter(
                            or_key,
                            chunk_prompt,
                            chunk_text,
                            model=or_model,
                        ),
                        batch_size=200,
                    )
                else:
                    refined = self._send_to_openrouter(or_key, prompt_for_file, file_text, model=or_model)
            else:
                self.status_cb(f"Nieobsługiwany tryb korekty: {corr_mode}", "warning")
                return "", gemini_rate_limited_until

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
                self.status_cb(
                    f"Korekta Gemini: przekroczony limit zapytań (429). Włączam cooldown ~{int(cooldown_s)}s.",
                    "warning",
                )
            else:
                if "openrouter" in mode:
                    provider_key = getattr(self.config, "openrouter_key", None)
                else:
                    provider_key = getattr(self.config, "gemini_key", None)
                self.status_cb(
                    f"Korekta ({ext.upper()}) nie powiodła się: {self._redact_api_key_in_message(str(e), provider_key)}",
                    "warning",
                )
            return "", gemini_rate_limited_until

        except Exception as e:
            if "openrouter" in mode:
                provider_key = getattr(self.config, "openrouter_key", None)
            else:
                provider_key = getattr(self.config, "gemini_key", None)
            self.status_cb(
                f"Korekta ({ext.upper()}) nie powiodła się: {self._redact_api_key_in_message(str(e), provider_key)}",
                "warning",
            )
            return "", gemini_rate_limited_until

        return str(refined or "").strip(), gemini_rate_limited_until

    def _redact_api_key_in_message(self, msg: str, api_key: str) -> str:
        try:
            if isinstance(api_key, str) and api_key:
                return str(msg).replace(api_key, "REDACTED")
        except Exception:
            pass
        return str(msg)

    def _format_eta(self, eta_seconds: float) -> str:
        if eta_seconds is None:
            return "--:--"
        if eta_seconds < 1:
            return "<1s"
        total = int(max(0, eta_seconds))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h:02}:{m:02}:{s:02}"
        return f"{m:02}:{s:02}"

    def _chunk_srt_segments(self, segments, max_items: int = 200, max_chars: int = None):
        chunks = []
        current = []
        current_chars = 0
        for seg in segments or []:
            txt = str(seg.get("text", "")).strip()
            add_len = len(txt) + 8
            if current and (len(current) >= max_items or (max_chars is not None and (current_chars + add_len) > max_chars)):
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(seg)
            current_chars += add_len
        if current:
            chunks.append(current)
        return chunks

    def _correct_srt_with_batched_provider(
        self,
        provider_name: str,
        prompt: str,
        file_segments: list,
        send_batch,
        batch_size: int = 200,
        inter_batch_sleep_s: float = 0.0,
    ):
        if not file_segments:
            return []

        chunks = self._chunk_srt_segments(file_segments, max_items=max(1, int(batch_size or 200)))
        try:
            if self.progress_cb:
                self.progress_cb(0)
        except Exception:
            pass

        corrected_lines = []
        started_at = time.time()
        total_segments = len(file_segments)
        for idx, chunk in enumerate(chunks):
            try:
                if self.status_cb:
                    self.status_cb(
                        f"Korekta {provider_name} SRT: paczka {idx + 1}/{len(chunks)} (segmentów: {len(chunk)})",
                        "info",
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

            response = send_batch(batch_prompt, "\n".join(numbered))
            parsed = parse_list_response(response)
            if not parsed:
                return None

            for i, seg in enumerate(chunk):
                if i < len(parsed) and str(parsed[i]).strip():
                    corrected_lines.append(str(parsed[i]).strip())
                else:
                    corrected_lines.append(str(seg.get("text", "")).strip())

            try:
                if self.progress_cb:
                    pct = int(((idx + 1) / max(1, len(chunks))) * 100)
                    self.progress_cb(max(0, min(100, pct)))
            except Exception:
                pass

            try:
                if self.status_cb and total_segments > 0:
                    processed = min(len(corrected_lines), total_segments)
                    elapsed = max(0.001, time.time() - started_at)
                    rate = processed / elapsed
                    remaining = max(0, total_segments - processed)
                    eta_seconds = (remaining / rate) if rate > 0 else None
                    self.status_cb(
                        f"Korekta {provider_name} SRT postęp: {processed}/{total_segments} segmentów | ETA: {self._format_eta(eta_seconds)}",
                        "info",
                    )
            except Exception:
                pass

            if inter_batch_sleep_s > 0 and idx < len(chunks) - 1:
                time.sleep(inter_batch_sleep_s)

        if len(corrected_lines) < len(file_segments):
            corrected_lines.extend([str(s.get("text", "")).strip() for s in file_segments[len(corrected_lines):]])
        return corrected_lines[:len(file_segments)]

    def _split_text_units_for_correction(self, text: str):
        src = str(text or "").strip()
        if not src:
            return []

        units = [p.strip() for p in re.split(r"\n\s*\n", src) if p and p.strip()]
        if len(units) <= 1:
            units = [ln.strip() for ln in src.splitlines() if ln and ln.strip()]

        if len(units) <= 1 and len(src) > 4000:
            sentence_parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", src) if s and s.strip()]
            if len(sentence_parts) > 1:
                units = sentence_parts

        return units if units else [src]

    def _chunk_text_units(self, units, max_items: int = 200, max_chars: int = 14000):
        chunks = []
        current = []
        current_chars = 0
        safe_max_items = max(1, int(max_items or 200))
        safe_max_chars = max(2000, int(max_chars or 14000))

        for unit in units or []:
            txt = str(unit).strip()
            if not txt:
                continue
            add_len = len(txt) + 2
            if current and (len(current) >= safe_max_items or (current_chars + add_len) > safe_max_chars):
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(txt)
            current_chars += add_len

        if current:
            chunks.append(current)
        return chunks

    def _correct_text_in_batched_chunks(self, file_text: str, base_prompt: str, provider_name: str, call_provider, batch_size: int = 200):
        units = self._split_text_units_for_correction(file_text)
        if not units:
            return ""

        chunks = self._chunk_text_units(units, max_items=batch_size)
        if len(chunks) <= 1:
            return call_provider(file_text, base_prompt)

        corrected_chunks = []
        processed_units = 0
        total_units = len(units)
        started_at = time.time()

        try:
            if self.progress_cb:
                self.progress_cb(0)
        except Exception:
            pass

        for idx, chunk_units in enumerate(chunks):
            chunk_text = "\n\n".join(chunk_units)
            prompt_for_chunk = (
                str(base_prompt or "").strip()
                + "\n\nINSTRUKCJA TECHNICZNA: To jest część "
                + f"{idx + 1}/{len(chunks)} długiego dokumentu. "
                + "Popraw tylko treść tej części, bez komentarzy i bez markdown."
            )

            try:
                if self.status_cb:
                    self.status_cb(
                        f"Korekta {provider_name} TXT/DOCX: paczka {idx + 1}/{len(chunks)} (jednostek: {len(chunk_units)})",
                        "info",
                    )
            except Exception:
                pass

            refined_chunk = str(call_provider(chunk_text, prompt_for_chunk) or "").strip()
            if not refined_chunk:
                refined_chunk = chunk_text
            corrected_chunks.append(refined_chunk)

            processed_units += len(chunk_units)
            try:
                if self.progress_cb and total_units > 0:
                    self.progress_cb(int((processed_units / total_units) * 100))
            except Exception:
                pass

            try:
                if self.status_cb and total_units > 0:
                    elapsed = max(0.001, time.time() - started_at)
                    rate = processed_units / elapsed
                    remaining = max(0, total_units - processed_units)
                    eta_seconds = (remaining / rate) if rate > 0 else None
                    self.status_cb(
                        f"Korekta {provider_name} TXT/DOCX postęp: {processed_units}/{total_units} | ETA: {self._format_eta(eta_seconds)}",
                        "info",
                    )
            except Exception:
                pass

        try:
            if self.progress_cb:
                self.progress_cb(100)
        except Exception:
            pass

        return "\n\n".join(part for part in corrected_chunks if str(part).strip()).strip()

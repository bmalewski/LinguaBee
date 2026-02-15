import json
import time
import re
import httpx

class OllamaRefiner:
    def __init__(self, ollama_model_name, status_callback=None):
        self.ollama_model_name = ollama_model_name
        self.status_callback = status_callback

    def _call_ollama_api(self, prompt_text):
        try:
            if self.status_callback:
                self.status_callback(f"Wysyłanie zapytania do Ollama (stream)...", "info")
            payload = {
                "model": self.ollama_model_name,
                "prompt": prompt_text,
                "stream": True
            }
            full_response = []
            # Use a reasonable timeout for refinement tasks; keep it bounded so failures surface quickly
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", "http://localhost:11434/api/generate", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            try:
                                response_json = json.loads(line)
                                response_part = response_json.get("response", "")
                                full_response.append(response_part)
                            except json.JSONDecodeError:
                                if self.status_callback:
                                    self.status_callback(f"Ostrzeżenie: Nie można zdekodować linii JSON ze strumienia Ollama: {line}", "warning")
            final = "".join(full_response).strip()
            if self.status_callback:
                self.status_callback(f"Zakończono refinowanie (stream). Otrzymano {len(final)} znaków.", "info")
            return final
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Błąd podczas wywołania API Ollama w refinerze: {repr(e)}", "error")
            raise

    def refine(self, text: str, custom_prompt: str = None) -> str:
        if not text or not text.strip():
            return text

        # If a custom prompt is provided by the user, use it. Otherwise use the default editor prompt.
        if custom_prompt and isinstance(custom_prompt, str) and custom_prompt.strip():
            prompt = custom_prompt.strip() + "\n\nTekst do poprawy:\n```\n" + text + "\n```"
        else:
            prompt = (
                "Jesteś doświadczonym redaktorem tekstu. Otrzymasz tekst już podzielony na akapity. "
                "Zachowaj istniejące akapity jeśli mają sens, popraw interpunkcję, popraw łamanie zdań i usuń ewidentne artefakty transkrypcji. "
                "Jeśli akapity łamią zdania w połowie, połącz je tak, aby każdy akapit kończył się pełnym zdaniem. "
                "Nie dodawaj nowych treści ani komentarzy. Zwróć tylko poprawiony tekst."
            )

            prompt = prompt + "\n\nTekst do poprawy:\n```\n" + text + "\n```"

        try:
            if self.status_callback:
                self.status_callback("Przygotowuję refinowanie transkryptu (chunking)...", "info")

            # Start with a smaller chunk size to reduce per-request generation time
            max_chunk_chars = 2000
            overlap = 300

            # If the input looks like a list of segments joined by a delimiter (e.g. '|||'),
            # prefer chunking by segment boundaries so we don't split segments in half.
            delim_pattern = r'\s*\|\|\|\s*'
            if re.search(r'\|\|\|', text):
                # split into logical segments, preserving order
                raw_segs = re.split(delim_pattern, text)
                # build pieces by grouping whole segments such that each piece <= max_chunk_chars
                pieces = []
                cur = []
                cur_len = 0
                for seg in raw_segs:
                    seg_str = seg.strip()
                    seg_len = len(seg_str)
                    # if single segment larger than max_chunk_chars, allow it as a single piece
                    if cur_len + seg_len + (3 if cur else 0) > max_chunk_chars and cur:
                        pieces.append(' ||| '.join(cur))
                        # start new bucket
                        cur = [seg_str]
                        cur_len = seg_len
                    else:
                        cur.append(seg_str)
                        cur_len += seg_len + (3 if cur_len else 0)
                if cur:
                    pieces.append(' ||| '.join(cur))
            else:
                pieces = []
                start = 0
                text_len = len(text)
                while start < text_len:
                    end = min(start + max_chunk_chars, text_len)
                    piece = text[start:end]
                    pieces.append(piece)
                    if end >= text_len:
                        break
                    start = end - overlap if (end - overlap) > start else end

            refined_pieces = []
            # retry policy
            max_retries = 2
            base_backoff = 1.0
            min_split_size = 600

            def try_refine_piece(piece_text, depth=0):
                """Try to refine a piece. If the first attempt times out and the piece is large, split it immediately.
                Otherwise perform a small number of retries before falling back."""
                # If pieces were created from segment groups joined by ' ||| ', preserve the surrounding
                # wrapper in the prompt but replace the large original block with the smaller piece.
                prompt_piece = prompt
                try:
                    # check for presence of a code fence in the prompt
                    if "```" in prompt and len(pieces) > 1:
                        prompt_piece = prompt.replace("```\n" + text + "\n```", "```\n" + piece_text + "\n```")
                    else:
                        # fallback: attach piece_text to end of prompt if no code fence
                        if len(pieces) > 1:
                            prompt_piece = prompt + "\n\nTekst do poprawy:\n```\n" + piece_text + "\n```"
                except Exception:
                    prompt_piece = prompt

                # First attempt: if it fails quickly (timeout or other), split early for large pieces
                try:
                    if self.status_callback:
                        self.status_callback(f"  -> wysyłam fragment do Ollama (pierwsza próba) (len={len(piece_text)})", "info")
                    return self._call_ollama_api(prompt_piece)
                except Exception as e_first:
                    if self.status_callback:
                        self.status_callback(f"  Pierwsza próba nieudana dla fragmentu (len={len(piece_text)}): {e_first}", "warning")
                    # If piece is large, split immediately to avoid long waits
                    if len(piece_text) > min_split_size and depth < 3:
                        if self.status_callback:
                            self.status_callback(f"  Fragment zbyt duży po pierwszej nieudanej próbie — dzielę (len={len(piece_text)})", "warning")
                        mid = len(piece_text) // 2
                        left = piece_text[:mid]
                        right = piece_text[mid:]
                        left_res = try_refine_piece(left, depth + 1)
                        right_res = try_refine_piece(right, depth + 1)
                        return (left_res or left).strip() + "\n\n" + (right_res or right).strip()

                # If we get here, perform limited retries for small pieces or non-timeout errors
                attempt = 1
                while attempt <= max_retries:
                    try:
                        if self.status_callback:
                            self.status_callback(f"  -> wysyłam fragment do Ollama (retry {attempt}/{max_retries}) (len={len(piece_text)})", "info")
                        return self._call_ollama_api(prompt_piece)
                    except Exception as e_retry:
                        if self.status_callback:
                            self.status_callback(f"  Retry {attempt} nieudany dla fragmentu (len={len(piece_text)}): {e_retry}", "warning")
                        time.sleep(base_backoff * (2 ** (attempt - 1)))
                        attempt += 1

                # Final fallback
                if self.status_callback:
                    self.status_callback(f"  Ostateczny fallback: używam oryginalnego fragmentu (len={len(piece_text)})", "warning")
                return piece_text

            for idx, piece in enumerate(pieces):
                if self.status_callback:
                    self.status_callback(f"Refinowanie fragmentu {idx+1}/{len(pieces)} (znaków: {len(piece)})...", "info")
                refined_chunk = try_refine_piece(piece, depth=0)
                refined_pieces.append(refined_chunk)
                time.sleep(0.2)

            final_refined = "\n\n".join([p.strip() for p in refined_pieces if p and p.strip()])
            if self.status_callback:
                self.status_callback(f"Skończono refinowanie. Otrzymano ~{len(final_refined)} znaków.", "info")
            return final_refined
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Refinement przez Ollama (refiner) nie powiódł się: {e}", "warning")
            raise

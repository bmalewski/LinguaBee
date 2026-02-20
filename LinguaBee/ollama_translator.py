import httpx
import json
import re

class OllamaTranslator:
    @staticmethod
    def check_connection():
        try:
            with httpx.Client(timeout=3) as client:
                response = client.get("http://localhost:11434/")
                response.raise_for_status()
            return True
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError):
            return False

    def __init__(self, ollama_model_name, status_callback=None):
        self.ollama_model_name = ollama_model_name
        self.status_callback = status_callback

    def _render_custom_prompt(self, custom_prompt, text, src_lang_full, tgt_lang_full):
        prompt = (custom_prompt or "").strip()
        if not prompt:
            return ""

        replacements = {
            "{src_lang}": src_lang_full,
            "{src_language}": src_lang_full,
            "{src_lang_full}": src_lang_full,
            "{tgt_lang}": tgt_lang_full,
            "{tgt_language}": tgt_lang_full,
            "{tgt_lang_full}": tgt_lang_full,
            "{text}": text,
        }
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)

        if "{text}" not in (custom_prompt or ""):
            prompt = f"{prompt}\n\nTekst:\n{text}"
        return prompt

    def _create_user_prompt_text(self, text, src_lang_full, tgt_lang_full, custom_prompt=None):
        custom_rendered = self._render_custom_prompt(custom_prompt, text, src_lang_full, tgt_lang_full)
        if custom_rendered:
            return custom_rendered
        # Use markers to avoid model adding commentary; translator usually returns plain text
        return f"""Jesteś ekspertem od tłumaczeń. Przetłumacz poniższy tekst z języka {src_lang_full} na język {tgt_lang_full}. Zwróć tylko i wyłącznie przetłumaczony tekst, bez żadnych dodatkowych zdań, komentarzy czy wyjaśnień.

Upewnij się, że odpowiedź nie zawiera wyjaśnień ani dodatkowego tekstu poza tłumaczeniem. Jeśli model ma tendencję do echo, postaraj się zwrócić sam tekst bez prefiksów.

Tekst do przetłumaczenia:
```
{text}
```"""

    def _create_user_prompt_batch(self, texts_json, src_lang_full, tgt_lang_full):
        return f"""Jesteś precyzyjnym asystentem tłumaczącym zoptymalizowanym do pracy z formatem JSON. Twoim zadaniem jest przetłumaczenie każdego tekstu z tablicy `texts_to_translate` z języka {src_lang_full} na język {tgt_lang_full}. Zwróć obiekt JSON zawierający wyłącznie klucz `translated_texts`, którego wartością jest tablica z przetłumaczonymi tekstami. Zachowaj tę samą kolejność. Nie dodawaj żadnych wyjaśnień ani dodatkowych komentarzy.

Oto dane wejściowe:
{texts_json}"""

    def _call_ollama_api(self, prompt_text):
        try:
            if self.status_callback:
                self.status_callback(f"Wysyłanie zapytania do Ollama (stream)...", "info")
            
            payload = {
                "model": self.ollama_model_name,
                "prompt": prompt_text,
                "stream": True,
                "temperature": 0.0,
                "max_tokens": 2048,
                "stop": ["<<<END_JSON>>>"]
            }

            full_response = []
            # Use a reasonable timeout per request (configurable if needed)
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", "http://localhost:11434/api/generate", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            try:
                                response_json = json.loads(line)
                                response_part = response_json.get("response", "")
                                full_response.append(response_part)
                                # stop early if translator returned end marker
                                if "<<<END_JSON>>>" in response_part:
                                    break
                            except json.JSONDecodeError:
                                if self.status_callback:
                                    self.status_callback(f"Ostrzeżenie: Nie można zdekodować linii JSON ze strumienia Ollama: {line}", "warning")

            final_translation = "".join(full_response).strip()
            if self.status_callback:
                self.status_callback(f"Zakończono tłumaczenie (stream). Otrzymano {len(final_translation)} znaków.", "info")

            return final_translation

        except httpx.ConnectError:
            if self.status_callback:
                self.status_callback("Błąd połączenia z serwerem Ollama. Upewnij się, że aplikacja Ollama jest uruchomiona.", "error")
            raise
        except httpx.ReadTimeout:
            if self.status_callback:
                self.status_callback("Przekroczono czas oczekiwania na odczyt podczas łączenia ze strumieniem Ollama.", "error")
            raise
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Błąd API Ollama: {repr(e)}", "error")
            raise

    def _call_ollama_api_batch(self, prompt_text):
        try:
            if self.status_callback:
                self.status_callback(f"Wysyłanie zapytania batch do Ollama (stream)... Prompt: {prompt_text[:200]}...", "info")
            
            payload = {
                "model": self.ollama_model_name,
                "prompt": prompt_text,
                "stream": True
            }

            full_response_str = ""
            # Batch requests can be long; allow more time but still bounded
            with httpx.Client(timeout=300.0) as client:
                with client.stream("POST", "http://localhost:11434/api/generate", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            try:
                                if self.status_callback:
                                    self.status_callback(f"Odebrano fragment odpowiedzi od Ollama: {line}", "info")
                                response_json = json.loads(line)
                                response_part = response_json.get("response", "")
                                full_response_str += response_part
                            except json.JSONDecodeError:
                                if self.status_callback:
                                    self.status_callback(f"Ostrzeżenie: Nie można zdekodować linii JSON ze strumienia Ollama: {line}", "warning")
            
            if self.status_callback:
                self.status_callback(f"Otrzymano odpowiedź JSON z Ollama: {full_response_str[:200]}...", "info")

            json_str = full_response_str
            
            # Try to find a markdown JSON block first
            match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_str)
            if match:
                json_str = match.group(1)
            else:
                # If no markdown block, try to find the first '{' and last '}'
                # This is a more robust way to extract JSON from conversational text
                start_idx = full_response_str.find('{')
                end_idx = full_response_str.rfind('}')
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    json_str = full_response_str[start_idx : end_idx + 1]
                else:
                    # If still no valid JSON structure found, log a warning and try to parse the whole thing
                    if self.status_callback:
                        self.status_callback(f"Ostrzeżenie: Nie znaleziono wyraźnego bloku JSON w odpowiedzi. Próba parsowania całej odpowiedzi.", "warning")
                    json_str = full_response_str # Fallback to parsing the whole string

            return json.loads(json_str)

        except httpx.ConnectError:
            if self.status_callback:
                self.status_callback("Błąd połączenia z serwerem Ollama. Upewnij się, że aplikacja Ollama jest uruchomiona.", "error")
            raise
        except httpx.ReadTimeout:
            if self.status_callback:
                self.status_callback("Przekroczono czas oczekiwania na odczyt podczas łączenia ze strumieniem Ollama.", "error")
            raise
        except json.JSONDecodeError:
            if self.status_callback:
                self.status_callback(f"Błąd krytyczny: Nie udało się sparsować ostatecznej odpowiedzi JSON z Ollama. Otrzymano: '{full_response_str}'.", "error")
            return {"translated_texts": []}
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Błąd API Ollama (batch): {repr(e)}", "error")
            raise
        
        return {"translated_texts": []}

    def translate(self, text, src_lang_full, tgt_lang_full, custom_prompt=None):
        if not text.strip():
            return ""
        prompt = self._create_user_prompt_text(text, src_lang_full, tgt_lang_full, custom_prompt=custom_prompt)
        return self._call_ollama_api(prompt)

    def translate_batch(self, texts: list, src_lang_full, tgt_lang_full, custom_prompt=None):
        if not texts:
            return []

        if custom_prompt and custom_prompt.strip():
            out = []
            for txt in texts:
                out.append(self.translate(txt, src_lang_full, tgt_lang_full, custom_prompt=custom_prompt))
            return out
        
        json_input = json.dumps({"texts_to_translate": texts}, ensure_ascii=False)
        prompt = self._create_user_prompt_batch(json_input, src_lang_full, tgt_lang_full)
        
        try:
            response_json = self._call_ollama_api_batch(prompt)
            return response_json.get("translated_texts", [])
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Błąd podczas batchowego tłumaczenia w Ollama: {repr(e)}", "error")
            return []

    def summarize(self, text, language):
        if not text.strip():
            return {{}}
        prompt = self._create_user_prompt_summary(text, language)
        
        original_callback = self.status_callback
        if original_callback:
            def summary_callback(msg, msg_type):
                msg = msg.replace("tłumaczenie", "streszczenie").replace("Tłumaczenie", "Streszczenie")
                original_callback(msg, msg_type)
            self.status_callback = summary_callback

        try:
            # We expect a JSON response, so we use the more robust _call_ollama_api_batch method
            # which is designed to extract JSON from potentially messy model outputs.
            response_json = self._call_ollama_api_batch(prompt)
            # Ensure the response has the expected keys
            if "propozycje" in response_json:
                return response_json
            else:
                if self.status_callback:
                    self.status_callback(f"Ostrzeżenie: Odpowiedź Ollama dla streszczenia ma nieoczekiwany format. Otrzymano: {response_json}", "warning")
                return {{}}
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Błąd podczas generowania streszczenia w Ollama: {repr(e)}", "error")
            return {{}}
        finally:
            # Restore original callback
            self.status_callback = original_callback
        
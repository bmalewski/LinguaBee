import httpx
import json
import re
import os
import datetime


class OllamaSummarizer:
    def __init__(self, ollama_model_name, status_callback=None):
        self.ollama_model_name = ollama_model_name
        self.status_callback = status_callback
        self.client = None  # Klient zostanie zainicjalizowany przy pierwszym użyciu

    def _initialize_client(self):
        """Inicjalizuje klienta i sprawdza dostępność modelu przy pierwszym użyciu."""
        if self.client is None:
            if self.status_callback:
                self.status_callback(f"Inicjalizacja klienta i weryfikacja modelu Ollama: {self.ollama_model_name}...", "info")
            self.client = httpx.Client(timeout=600.0)

    def _call_ollama_api(self, prompt_text):
        try:
            if self.status_callback:
                self.status_callback("Wysyłanie zapytania o streszczenie do Ollama...", "info")

            payload = {
                "model": self.ollama_model_name,
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p": 0.9,
                    "num_predict": 1200,
                    "repeat_penalty": 1.05
                }
            }

            resp = self.client.post("http://localhost:11434/api/generate", json=payload)
            resp.raise_for_status()
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            model_text = (body.get("response") or "").strip() if isinstance(body, dict) else ""

            # Respect user prompt output as-is. If it's valid JSON, return parsed dict; otherwise raw text.
            parsed = self._extract_json_candidate(model_text)
            return parsed if parsed is not None else model_text

        except httpx.ConnectError as e:
            if self.status_callback:
                self.status_callback("Błąd połączenia z serwerem Ollama. Upewnij się, że aplikacja Ollama jest uruchomiona.", "error")
            raise e
        except httpx.ReadTimeout as e:
            if self.status_callback:
                self.status_callback("Przekroczono czas oczekiwania na odczyt podczas łączenia z Ollama.", "error")
            raise e
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Błąd API Ollama (streszczenie): {repr(e)}", "error")
            raise e

    def _extract_json_candidate(self, text):
        if not text:
            return None
        t = text.strip()
        # remove code fences
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
        # direct parse
        try:
            data = json.loads(t)
            if isinstance(data, dict):
                if "propozycje" in data:
                    return data
                if all(k in data for k in ["formalne", "intrygujace", "zabawne"]):
                    return {"propozycje": data}
        except Exception:
            pass
        # find first JSON object
        m = re.search(r"(\{[\s\S]*\})", t)
        if m:
            try:
                data = json.loads(m.group(1))
                if isinstance(data, dict):
                    if "propozycje" in data:
                        return data
                    if all(k in data for k in ["formalne", "intrygujace", "zabawne"]):
                        return {"propozycje": data}
            except Exception:
                return None
        return None

    def _normalize_language(self, language):
        """Normalize language input to a human-readable name and ISO code."""
        if not language:
            return ("English", "en")
        lang = str(language).strip()
        key = lang.lower()
        mapping = {
            "pl": ("Polski", "pl"),
            "polish": ("Polski", "pl"),
            "polski": ("Polski", "pl"),
            "en": ("English", "en"),
            "english": ("English", "en"),
            "de": ("Deutsch", "de"),
            "deutsch": ("Deutsch", "de"),
            "fr": ("Français", "fr"),
            "francais": ("Français", "fr"),
            "es": ("Español", "es"),
            "spanish": ("Español", "es"),
        }
        return mapping.get(key, (lang, key))

    def _plaintext_to_propozycje(self, text):
        # Return default empty structure when there's no text
        if not text or not text.strip():
            return {
                "propozycje": {
                    "formalne": {"tytul": "", "opis": ""},
                    "intrygujace": {"tytul": "", "opis": ""},
                    "zabawne": {"tytul": "", "opis": ""}
                }
            }

        lowered = text.lower()
        sections = {}

        # First try: detect explicit header blocks like '=== NAME ===' or '### NAME ###'
        header_pattern = re.compile(r"(^|\n)\s*(?:=+|#{2,})\s*([^=#\n]+?)\s*(?:=+|#{2,})\s*(?:\n|$)", flags=re.I)
        headers = list(header_pattern.finditer(text))
        if headers:
            # If headers found, extract text between headers
            for i, m in enumerate(headers):
                header_name = m.group(2).strip().lower()
                start = m.end()
                end = len(text)
                if i + 1 < len(headers):
                    end = headers[i+1].start()
                block = text[start:end].strip()
                # map header_name to one of our keys
                if any(k in header_name for k in ["formal", "formale", "formalne", "oficjal"]):
                    sections.setdefault("formalne", "")
                    sections["formalne"] = block
                elif any(k in header_name for k in ["intryg", "intrigu", "zaczep", "hook", "intrig"]):
                    sections.setdefault("intrygujace", "")
                    sections["intrygujace"] = block
                elif any(k in header_name for k in ["zabaw", "funny", "humor", "lustig", "witzig"]):
                    sections.setdefault("zabawne", "")
                    sections["zabawne"] = block
                else:
                    # If header name unknown, attempt to place into formal if empty
                    if "formalne" not in sections:
                        sections["formalne"] = block

        # If explicit headers not found or didn't yield sections, fallback to keyword markers
        if not sections:
            # Candidate markers for each category (Polish/English/German)
            markers = {
                "formalne": ["formalne", "formal", "formale", "oficjalne", "formalna"],
                "intrygujace": ["intrygujace", "intrygujące", "intrygujący", "intriguing", "hook", "zaczepne", "intritt", "intrig"],
                "zabawne": ["zabawne", "funny", "humorous", "ciekawe", "lustig", "witzig"]
            }

            # Find positions of markers
            positions = {}
            for key, kws in markers.items():
                pos = -1
                for kw in kws:
                    idx = lowered.find(kw)
                    if idx != -1 and (pos == -1 or idx < pos):
                        pos = idx
                if pos != -1:
                    positions[key] = pos

            # If no markers found, return everything as formal description
            if not positions:
                return {
                    "propozycje": {
                        "formalne": {"tytul": "Streszczenie", "opis": text.strip()},
                        "intrygujace": {"tytul": "", "opis": ""},
                        "zabawne": {"tytul": "", "opis": ""}
                    }
                }

            # Sort found positions and split
            sorted_keys = sorted(positions.items(), key=lambda x: x[1])
            for i, (key, pos) in enumerate(sorted_keys):
                start = pos
                end = len(text)
                if i + 1 < len(sorted_keys):
                    end = sorted_keys[i+1][1]
                sections[key] = text[start:end].strip()

        def extract_title_and_desc(block):
            if not block:
                return ("", "")

            # Preliminary cleaning: remove obvious decorative tokens and normalize separators
            cleaned = re.sub(r"---\s*Opis\s*---", "", block, flags=re.I)
            cleaned = re.sub(r"[=\-]{2,}", "\n", cleaned)

            # Preserve category words but strip leading/trailing decorative characters from each line
            cleaned_lines = []
            for l in cleaned.splitlines():
                s = l.strip()
                if not s:
                    continue
                s = re.sub(r"^[=\-\s]+", "", s)
                s = re.sub(r"[=\-\s]+$", "", s)
                s = s.strip()
                if not s:
                    continue
                cleaned_lines.append(s)

            cleaned = "\n".join(cleaned_lines).strip()

            # Match common variants (Polish with/without diacritics, English) on cleaned text
            m_title = re.search(r"(?:tytuł|tytul|title)\s*[:\-]\s*(.+)", cleaned, flags=re.I | re.UNICODE)
            m_desc = re.search(r"(?:opis|description)\s*[:\-]\s*(.+)", cleaned, flags=re.I | re.S | re.UNICODE)
            if m_title and m_desc:
                return (m_title.group(1).strip(), m_desc.group(1).strip())

            # If labeled forms not found, heuristic fallback: first meaningful line as title
            lines = [l for l in cleaned.splitlines() if l.strip()]
            if not lines:
                return ("", "")
            if len(lines) == 1:
                return ("", lines[0])
            title = lines[0]
            desc = "\n".join(lines[1:]).strip()

            # If the description still contains a labeled 'Opis:' later, prefer the labeled content
            m_desc_in_desc = re.search(r"(?:opis|description)\s*[:\-]\s*(.+)", desc, flags=re.I | re.S | re.UNICODE)
            if m_desc_in_desc:
                desc = m_desc_in_desc.group(1).strip()

            return (title, desc)

        propozycje = {}
        for k in ["formalne", "intrygujace", "zabawne"]:
            t, d = extract_title_and_desc(sections.get(k, ""))
            propozycje[k] = {"tytul": t, "opis": d}
        return {"propozycje": propozycje}

    def _validate_propozycje(self, data):
        """Validate that the parsed object contains three categories and non-empty content.

        Returns True if the structure looks valid enough to use.
        """
        if not isinstance(data, dict):
            return False
        propo = data.get("propozycje") if isinstance(data, dict) else None
        if not isinstance(propo, dict):
            return False
        keys = ["formalne", "intrygujace", "zabawne"]
        for k in keys:
            v = propo.get(k)
            if not isinstance(v, dict):
                return False
            t = (v.get("tytul") or "").strip()
            d = (v.get("opis") or "").strip()
            # At least one of title or description should be non-empty
            if not t and not d:
                return False
        return True

    def _create_user_prompt_summary(self, text, language, custom_prompt=None):
        language_name, language_code = self._normalize_language(language)
        if self.status_callback:
            self.status_callback(f"Tworzę prompt dla języka: {language_name} ({language_code})", "debug")

        language_lock = (
            "WAŻNE: Zwróć odpowiedź WYŁĄCZNIE w języku "
            f"{language_name} (kod: {language_code}). "
            "Nie używaj innego języka i nie mieszaj języków."
        )

        if custom_prompt and isinstance(custom_prompt, str) and custom_prompt.strip():
            if self.status_callback:
                self.status_callback("Używam niestandardowego promptu użytkownika dla streszczenia Ollama.", "info")
            p = custom_prompt.strip()
            if "{text}" in p:
                p = p.replace("{text}", text)
            else:
                p = p + "\n\nTekst do analizy:\n" + text
            p = p.replace("{language_name}", language_name).replace("{language_code}", language_code)
            return language_lock + "\n\n" + p

        # Brak promptu domyślnego: używamy WYŁĄCZNIE promptu użytkownika z okna dialogowego.
        return language_lock + "\n\nTekst do analizy:\n" + text

    def summarize(self, text, language, custom_prompt=None):
        """Generate summary proposals using Ollama and optionally translate results if they're in English.

        Returns a dict containing the 'propozycje' key on success, or an empty dict on failure.
        """
        if not text or not text.strip():
            return {}
        self._initialize_client()
        prompt = self._create_user_prompt_summary(text, language, custom_prompt=custom_prompt)
        try:
            # Zwracamy wynik dokładnie z modelu (tekst lub JSON),
            # aby respektować prompt użytkownika z dialogu.
            return self._call_ollama_api(prompt)
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Błąd podczas wywołania Ollama (streszczenie): {e}", "error")
            return {}

# LinguaBee 0.9.9

## Krótki opis (do GitHub Release)

LinguaBee 0.9.9 to wydanie porządkujące i stabilizujące. Naprawia dwa poważne błędy (maskowanie wyjątków oraz niedziałające przetwarzanie audio), usuwa martwy kod i niespójności, a także przywraca pełną funkcjonalność limitu linii w napisach SRT oraz poprawia obsługę znaków diakrytycznych.

## Najważniejsze zmiany

### Naprawione błędy krytyczne

- **Maskowanie wyjątków w `worker.py`:** funkcja `_redact_api_key_in_message(...)` była wywoływana podczas obsługi błędów, ale nigdy nie była zaimportowana — każdy wyjątek kończył się dodatkowym `NameError`, ukrywając prawdziwą przyczynę problemu. Funkcja jest teraz prawidłowo importowana z `correction_service`.
- **Niedziałające przetwarzanie audio:** `worker.py` uruchamiał izolowany subprocess `tools/process_audio_runner.py`, który w ogóle nie istniał w repozytorium. W praktyce odszumianie, normalizacja i konwersja do mono nigdy się nie wykonywały (cichy fallback na oryginalny plik). Dodano brakujący runner, który w izolowanym procesie wywołuje `process_audio()` z `audio_processing.py`.

### Przywrócona i ulepszona funkcjonalność

- **Limit linii w napisach SRT (`max_lines`) znów działa:** parametr był przekazywany z GUI aż do `save_srt`, lecz był ignorowany — zbyt długie wypowiedzi trafiały do jednego napisu. Teraz, gdy zawinięty tekst przekracza ustawiony limit linii, segment jest dzielony na kolejne napisy z **proporcjonalnym podziałem czasu trwania** (bez powrotu do sztywnego łamania znaczników czasowych wyrazów).
- **Poprawna obsługa znaków diakrytycznych w `add_missing_spaces`:** dotychczasowy wzorzec obejmował wyłącznie litery ASCII (`A–Z`), przez co nie wstawiał brakującej spacji przed słowami z polskimi/akcentowanymi znakami (np. `koniec.Ąby`). Zastąpiono go klasami Unicode — liczby (np. `1.23`) i nawiasy nadal są obsługiwane poprawnie.

### Porządki i spójność

- Usunięto martwe pliki:
  - `whisper_transcriptionท.py` — przestarzała kopia zapasowa z błędnym znakiem Unicode w nazwie (aktualna, lepsza wersja znajduje się w `whisper_transcription.py`),
  - `summarizer.py` — pusty, nieużywany.
- `worker.py`: usunięto zduplikowaną definicję `_make_ollama_status_cb` oraz nieużywane importy (`httpx`, `redistribute_text_to_segments`).
- `main.py`: aktywowano dotychczas nieużywaną funkcję `_load_custom_fonts()` (bezpieczne wczytywanie czcionek projektu, np. SF Pro, jeśli katalog `font/` istnieje).
- `config.py`: dodano brakujące pola `audio_processing_timeout` i `gemini_cooldown_seconds`, do których kod odwoływał się dotąd wyłącznie przez `getattr` z wartościami domyślnymi.
- `.gitignore`: dodano wyjątek dla `tools/process_audio_runner.py`, aby ten plik produkcyjny był wersjonowany mimo ignorowania reszty katalogu `/tools`.

## Zmienione pliki

- `worker.py`
- `file_utils.py`
- `text_utils.py`
- `config.py`
- `main.py`
- `.gitignore`
- `tools/process_audio_runner.py` (nowy)
- usunięto: `whisper_transcriptionท.py`, `summarizer.py`
- `setup.iss` (AppVersion 0.9.9)

## Tag

- Tag: `v0.9.9`

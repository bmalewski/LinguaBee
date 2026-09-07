# LinguaBee 1.0.1

## Krótki opis (do GitHub Release)

LinguaBee 1.0.1 to wydanie naprawcze po pełnym przeglądzie kodu pod Pythonem 3.12. Usuwa cztery błędy blokujące funkcje (diaryzacja, przycisk Stop, logowanie do Hugging Face, obsługa błędów PyAV), naprawia rozjazd segmentów przy korekcie SRT, sześć błędów interfejsu oraz szereg problemów w backendach tłumaczeń, korekty i streszczeń. Nie wprowadza nowych funkcji.

## Najważniejsze zmiany

### Naprawione błędy krytyczne

- **Diaryzacja zawsze kończyła się błędem:** zainstalowany `pyannote.audio` 4.x zwraca obiekt `DiarizeOutput` bez metody `itertracks`, przez co każda próba rozpoznania mówców kończyła się wyjątkiem i cichym pominięciem. `audio_processing.py` rozpoznaje teraz nowy format (adnotacja w polu `speaker_diarization`) i nadal obsługuje starsze wersje 3.x.
- **Przycisk Stop podczas transkrypcji zgłaszał fałszywy błąd:** `WhisperTranscription.transcribe()` zwracała przy zatrzymaniu dwie wartości, a `worker.py` rozpakowywał trzy (`ValueError`). Wszystkie ścieżki zatrzymania zwracają teraz trójkę.
- **Logowanie do Hugging Face nigdy się nie wykonywało:** `huggingface_utils.py` używał niezaimportowanej klasy `HfFolder` (`NameError` połykany przez `try/except`). Zastąpiono ją funkcją `get_token()` z `huggingface_hub`.
- **Zepsuta obsługa błędów ekstrakcji audio z wideo:** `file_utils.py` łapał `av.AVError`, którego PyAV 14+ już nie ma, więc każdy błąd dekodowania zamieniał się w mylący `AttributeError`. Klasa wyjątku jest teraz wybierana przy imporcie (`av.error.FFmpegError` w nowych wersjach).

### Korekta SRT

- **Rozjazd segmentów względem znaczników czasu:** adaptery Gemini/OpenRouter zwracały listę poprawionych segmentów sklejoną `"\n\n"`, a `correction_service` parsował ją ponownie linia po linii. Ponieważ plik SRT jest zapisywany z zawijaniem do N znaków, kwestie wieloliniowe rozpadały się na kilka elementów i cały plik przesuwał się od drugiej kwestii w dół. Tekst segmentów jest teraz spłaszczany przed wysłaniem do modelu, a wynik wraca jako lista JSON odtwarzana 1:1.
- **Korekta SRT przez Ollama poprawiała tylko pierwsze ~2000 znaków:** cała lista przechodziła przez wewnętrzny chunking `OllamaRefiner`, a parser brał tylko pierwszą listę JSON. Ollama korzysta teraz z tego samego trybu wsadowego po segmentach, co Gemini i OpenRouter.
- **Jedna nieczytelna paczka nie kasuje już wyników poprzednich:** segmenty z takiej paczki zachowują oryginalny tekst, a w logu pojawia się ostrzeżenie (dotychczas cała korekta zwracała pusty wynik).
- **Ostrzeżenie o niezgodnej liczbie segmentów** zamiast cichego dopełniania oryginałem.
- **Wynik korekty TXT/DOCX trafia do tłumaczenia i streszczenia:** dotąd tylko korekta SRT aktualizowała tekst w pipeline.
- **Korekta bez zaznaczonych formatów oryginalnych** loguje ostrzeżenie z wyjaśnieniem zamiast pomijać się bez śladu.
- **Ustawienia zawijania SRT (`max_lines`, `max_chars_per_line`)** są respektowane także dla plików `_translation.srt` i `_corrected.srt`, nie tylko `_original.srt`.

### Interfejs (`gui/main_window.py`, `gui/dialogs.py`)

- **Wybór pliku lokalnego po wpisaniu URL kasował wybór:** czyszczenie pola URL wyzwalało `textChanged` → `clear_local_files`. Pole jest teraz czyszczone z zablokowanymi sygnałami.
- **Pierwsze uruchomienie odznaczało wszystkie formaty wyjściowe:** przy braku `user_settings.json` (lub starym pliku bez tych kluczy) zachowywane są domyślne zaznaczenia widgetu.
- **Stop → Start mógł zabić aplikację:** Stop tylko ustawiał flagę, a Start nadpisywał referencję do wciąż działającego `QThread` ("QThread: Destroyed while thread is still running"). Referencja nazywa się teraz `worker_thread` (usunięta kolizja z `QObject.thread()`), Start pozostaje zablokowany do faktycznego zakończenia wątku, a próba startu w trakcie kończenia pokazuje ostrzeżenie.
- **Zamykanie okna w trakcie zadania:** dodano `closeEvent`, który pyta o potwierdzenie, zatrzymuje wątek i czeka na niego.
- **Zatrzymanie przez użytkownika** pokazuje okno informacyjne, a nie czerwone "Błąd".
- **`preset_group` dodawany do dwóch layoutów** (ostrzeżenie Qt) — usunięto.
- **`OllamaSettingsDialog`** wywoływał nieistniejącą metodę `populate_models` — dodano ją wraz z `get_settings()`.

### Backendy i API

- **Ollama (tłumaczenie):** `temperature`, limit tokenów i `stop` były wysyłane na najwyższym poziomie payloadu, gdzie Ollama je ignoruje. Przeniesiono do `options` (`num_predict`). Tłumaczenie honoruje też jawnie ustawiony język źródłowy tłumaczenia, jak pozostałe backendy. Usunięto martwą metodę `summarize()` odwołującą się do metody z innej klasy.
- **Ollama (refinowanie / korekta TXT):** fragmenty nakładały się o 300 znaków, a nakładka nie była usuwana po złączeniu (duplikacja tekstu na każdej granicy). Przy podziale po timeoucie obie połówki dostawały pełny tekst (wynik zdublowany). Chunking działa teraz bez nakładania, z cięciem na granicy akapitu lub zdania, a każdy fragment dostaje wyłącznie własny tekst.
- **Ollama (streszczenia):** `num_predict` podniesiony z 1200 do 4096 — poprzedni limit ucinał wyniki dostarczonych promptów (np. Social Media).
- **Gemini (`api_client.py`):** klucz API przekazywany w nagłówku `x-goog-api-key` zamiast w URL (komunikaty błędów httpx zawierają pełny URL). Zdjęto sztywny limit `maxOutputTokens: 4096`, który obcinał paczki korekty i streszczenia (modele z "thinking" liczą do niego również tokeny rozumowania). Odpowiedź z `finishReason: MAX_TOKENS` zgłasza błąd zamiast zapisywać się jako pełna.
- **OpenRouter (tłumaczenie SRT):** przy własnym prompcie numerowana lista segmentów była osadzana w prompcie dwukrotnie, co często skutkowało podwojoną liczbą odpowiedzi i fallbackiem na pojedyncze wywołania.
- **NLLB:** fragmenty tłumaczenia sklejane pustym ciągiem zamiast spacją, co zlepiało słowa na granicach.
- **BART:** tokenizer używał `max_length=2048` przy modelu o 1024 pozycjach (błąd indeksu w osadzeniach pozycyjnych przy długich wejściach → puste streszczenie). Limit pobierany z konfiguracji modelu.
- **yt-dlp:** błędna nazwa opcji `no_check_certificate` → `nocheckcertificate`.
- **WhisperX:** `whisperx.align()` zwraca słownik `{"segments", "word_segments"}`, a nie listę — `whisper_transcription.py` rozpakowuje go poprawnie. `whisper_aligner.py` wywołuje `align()` z właściwą sygnaturą (osobny model wyrównania) i czyta `word_segments`.

### Porządki

- **Pliki tymczasowe audio:** po odszumianiu/normalizacji oryginalny plik pobrany z YouTube (lub wyekstrahowany z wideo) nie był kasowany mimo `delete_audio`, bo ścieżka do usunięcia była nadpisywana ścieżką pliku przetworzonego. Worker pamięta teraz listę plików do sprzątnięcia.

## Środowisko

- Kod zweryfikowany pod Pythonem 3.12.14 (kompilacja z `SyntaxWarning` jako błędem, brak usuniętych modułów stdlib).
- Uwaga dla Apple Silicon: po odtworzeniu `venv` na Pythonie 3.12 backend MLX wymaga ręcznie `pip install mlx-lm` (patrz `requirements.txt`).

## Zmienione pliki

- `api_client.py`
- `audio_processing.py`
- `bart_summarizer.py`
- `correction_adapters.py`
- `correction_service.py`
- `downloader.py`
- `file_utils.py`
- `gui/dialogs.py`
- `gui/main_window.py`
- `huggingface_utils.py`
- `ollama_refiner.py`
- `ollama_summarizer.py`
- `ollama_translator.py`
- `translation_manager.py`
- `whisper_aligner.py`
- `whisper_transcription.py`
- `worker.py`
- `RELEASE_NOTES_1.0.1.md` (nowy)
- `setup.iss` (AppVersion 1.0.1)

## Tag

- Tag: `v1.0.1`

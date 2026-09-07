# LinguaBee 1.0.2

## Krótki opis (do GitHub Release)

LinguaBee 1.0.2 dodaje budowanie samodzielnej aplikacji macOS (`LinguaBee.app`, Apple Silicon) uruchamianej dwuklikiem, naprawia diaryzację, która nie działała bez bibliotek współdzielonych ffmpeg, oraz porządkuje katalogi zapisu danych w spakowanej aplikacji. Backend MLX nie jest już instalowany domyślnie z powodu konfliktu zależności z whisperx.

## Najważniejsze zmiany

### Aplikacja macOS (.app)

- **Nowe pliki `packaging/LinguaBee-macos.spec` i `packaging/build_macos.sh`:** budowanie przez PyInstaller (one-dir, arm64), z wbudowanymi statycznymi binarkami `ffmpeg`/`ffprobe`, ikoną `.icns` generowaną z `icons/`, podpisem ad-hoc i opcjonalną instalacją do `/Applications` (`--install`). Gotowa aplikacja ma ok. 1,1 GB; modele pobierają się przy pierwszym użyciu, jak dotąd.
- **Katalogi danych w spakowanej aplikacji** (`config.py`): wyniki trafiają do `~/Documents/LinguaBee`, a modele, `user_settings.json` i szablony promptów do `~/Library/Application Support/LinguaBee` (Windows: `%APPDATA%\LinguaBee`). Dotychczas wszystko było liczone względem pliku wykonywalnego, co w pakiecie `.app` oznaczało zapis do jego wnętrza. Uruchomienie ze źródeł (`python main.py`) zachowuje katalogi `output/`, `models/`, `prompts/` obok `main.py`.
- **Zasoby tylko do odczytu** (ikony, `gui/stylesheet.qss`, seedy promptów, wbudowane binarki) są czytane przez `config.resource_path()`, które w spakowanej aplikacji wskazuje do `sys._MEIPASS`. Dołączone szablony promptów są kopiowane do katalogu użytkownika przy pierwszym uruchomieniu (bez nadpisywania).
- **Izolowany runner audio w spakowanej aplikacji:** `worker.py` uruchamiał `sys.executable` ze skryptem `.py`, co w `.app` otwierałoby drugie okno programu. `main.py` obsługuje teraz tryb `--audio-runner` przed importem Qt, a `worker.py` używa go, gdy aplikacja jest spakowana.
- **Tryb `--selftest`** w `main.py`: importuje wszystkie leniwie ładowane biblioteki i sprawdza dostępność `ffmpeg`/`ffprobe`; służy do weryfikacji zbudowanej aplikacji.
- **Wbudowane `ffmpeg`/`ffprobe` i Homebrew na PATH:** aplikacja uruchomiona dwuklikiem dziedziczy minimalny PATH launchd, więc `main.py` dopisuje katalog wbudowanych binarek oraz `/opt/homebrew/bin` i `/usr/local/bin`. Cache JIT numba jest kierowany do `~/Library/Caches/LinguaBee`.
- `gui/dialogs.py`: siedem miejsc liczących katalog `prompts/` względem `__file__` korzysta z `config.prompts_subdir()`.
- `tools/__init__.py`: katalog `tools` jest teraz pakietem (wymagane przez analizę importów PyInstallera).
- README: nowa sekcja „9) Budowanie aplikacji .app na macOS (Apple Silicon)”.

### Naprawiona diaryzacja

- **Diaryzacja nie działała bez bibliotek współdzielonych ffmpeg:** pyannote.audio 4 dekoduje pliki przez `torchcodec`, który wymaga `libav*.dylib` (Homebrew), niedostępnych na wielu instalacjach i w spakowanej aplikacji. `audio_processing.py` dekoduje teraz audio przez PyAV (`faster_whisper.audio.decode_audio`) i przekazuje pyannote gotową falę w pamięci (`{"waveform", "sample_rate"}`), czyli oficjalnie wspieraną ścieżkę wejścia. W razie niepowodzenia dekodowania następuje powrót do ścieżki pliku.

### Zależności

- **Backend MLX (mlx-lm) nie jest instalowany domyślnie:** `mlx-lm >= 0.30` wymaga transformers 5 i huggingface-hub 1.x, a whisperx 3.8 wymaga huggingface-hub < 1.0. Opcja „MLX Apple” pozostaje w GUI, ale profil sprzętowy rekomenduje ją tylko, gdy pakiet jest zainstalowany. Instrukcja ręcznej instalacji (`pip install "mlx-lm<0.30"`) jest w `requirements.txt`.
- `.gitignore`: ignorowane `build/`, `dist/`, `build_assets/`; wersjonowany `tools/__init__.py`.

## Zmienione pliki

- `config.py`
- `main.py`
- `worker.py`
- `audio_processing.py`
- `gui/dialogs.py`
- `tools/process_audio_runner.py`
- `tools/__init__.py` (nowy)
- `packaging/LinguaBee-macos.spec` (nowy)
- `packaging/build_macos.sh` (nowy)
- `requirements.txt`
- `README.md`
- `.gitignore`
- `RELEASE_NOTES_1.0.2.md` (nowy)
- `setup.iss` (AppVersion 1.0.2)

## Tag

- Tag: `v1.0.2`

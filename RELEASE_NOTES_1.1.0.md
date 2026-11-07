# LinguaBee 1.1.0

## Krótki opis (do GitHub Release)

LinguaBee 1.1.0 wprowadza lekki instalator dla Windows w stylu ComfyUI Desktop. `LinguaBee-1.1.0-Setup.exe` ma ok. 20 MB i sam pobiera Pythona, PyTorch dla wybranej karty, biblioteki i ffmpeg do jednego folderu na dowolnym dysku. Program uruchamia się jednym kliknięciem: ze skrótu albo pliku `LinguaBee.bat`, bez wpisywania poleceń w PowerShellu. Start jest też szybszy: ekran startowy pojawia się po ok. 1 s, a ciężkie biblioteki ładują się w tle.

## Najważniejsze zmiany

### Lekki instalator Windows (`packaging/installer/`)

- **Wybór folderu i karty:** folder może być na dowolnym dysku, poza Program Files i OneDrive. Karta: NVIDIA, AMD (eksperymentalnie) albo tylko procesor; instalator wykrywa ją sam. Nie wymaga uprawnień administratora.
- **Pobieranie:**
  - Python 3.12 (uv, python-build-standalone);
  - biblioteki z przypiętych list z sumami SHA-256 (`requirements/windows-nvidia.txt`, `windows-cpu.txt`, `windows-amd.txt`);
  - ffmpeg;
  - opcjonalnie Deno do YouTube.
- **Rozmiar:** NVIDIA to ok. 4 GB do pobrania i ok. 8,5 GB na dysku, wariant CPU ok. 1 GB i ok. 4,5 GB. Modele dochodzą osobno.
- **Wszystko w jednym folderze:** `app`, `python`, `bin`, `models`, `output`, `user` (ustawienia, prompty, logi) i `cache`. Folder można przenieść na inny dysk; potem wystarczy uruchomić instalator i wskazać nowe miejsce.
- **„Napraw instalację”:**
  - sprawdza pliki bibliotek z zapisanymi rozmiarami (`verify_packages.py`) i instaluje ponownie uszkodzone;
  - uzupełnia braki;
  - po przeniesieniu folderu poprawia skróty.
- **Aktualizacja:** wystarczy uruchomić nowszy instalator w tym samym folderze. Pobiera tylko zmiany, a yt-dlp zawsze w najnowszej wersji.
- **Deinstalacja:** usuwa program, Pythona i biblioteki. Pyta, czy usunąć także modele, wyniki i ustawienia; domyślnie zostają.
- **Budowanie:** instalator buduje `build_installer.ps1`. Bierze kod z `git ls-files`, więc nie trafiają do niego np. `user_settings.json` z kluczami API, modele ani wyniki. Listy pakietów generuje `compile_locks.ps1`.

### Start jednym kliknięciem

- `LinguaBee.bat` uruchamia program bez konsoli, `LinguaBee - konsola.bat` z oknem komunikatów do diagnozy.
- `Utworz skrot na pulpicie.bat` tworzy skrót z ikoną.
- `setup_windows.bat` instaluje ze źródeł bez wpisywania poleceń w PowerShellu.
- Start bez konsoli zapisuje komunikaty do `logs\linguabee.log` (w instalacji: `user\logs`). Błąd przy starcie pokazuje okno z informacją, gdzie szukać szczegółów.

### Szybszy start programu

- **Ekran startowy** z logo i bieżącym etapem pojawia się po ok. 1,3 s od kliknięcia.
- **Okno pojawia się po ok. 3,7 s zamiast 6,9 s.** Pomiar na RTX 3090 Ti, gdy biblioteki są już w pamięci podręcznej. Po dłuższej przerwie stara wersja potrzebowała ok. 45 s i przez ten czas nic nie było widać.
- **Biblioteki w tle:** przed oknem ładuje się tylko PyTorch, potrzebny do wykrycia karty. Whisper, transformers, yt-dlp, PyAV i python-docx ładują się w tle, gdy okno już działa (`worker.load_backends`). Start kliknięty wcześniej czeka na nie z komunikatem w dzienniku.
- **AMD:** lista kart AMD jest odczytywana w tle, gdy widać ekran startowy.
- **`--selftest`** sprawdza też moduły programu, bo nie ładują się już przy starcie. Instalator uruchamia go na koniec instalacji, więc ich pliki .pyc są gotowe przed pierwszym startem.

### Tryb instalacji w kodzie

- **`config.py`:**
  - rozpoznaje instalację po `install.json` obok folderu `app` albo po zmiennej `LINGUABEE_HOME`;
  - w instalacji modele, wyniki, ustawienia i cache (Hugging Face, torch, yt-dlp, numba, matplotlib) trafiają do jej folderu, a `bin` jest na PATH;
  - `install.json` może wskazać inne miejsce na modele i wyniki (`models_dir`, `output_dir`);
  - `APP_VERSION` jest jedynym źródłem numeru wersji;
  - wyłącza statystyki pyannote (`PYANNOTE_METRICS_ENABLED=false`).
- **`huggingface_utils.py`:** token Hugging Face jest w folderze z `HF_HOME`, a w instalacji w `user\huggingface`. Token z dawnych lokalizacji nadal działa. Obsługuje też zmienną `HF_TOKEN`.
- **`requirements.txt`:** `PySide6-Essentials` zamiast `PySide6`. Program używa tylko QtCore, QtGui i QtWidgets, więc to ok. 400 MB mniej.
- **Usunięty `setup.iss`:** dawny instalator jednego pliku .exe z wersji 1.0.x.

## Uwagi

- Instalator nie jest podpisany cyfrowo, więc Windows SmartScreen pokaże ostrzeżenie: „Więcej informacji” → „Uruchom mimo to”.
- Wariant AMD (ROCm) jest eksperymentalny i nie był jeszcze testowany z tym instalatorem na karcie AMD.
- Dla kart NVIDIA zalecany jest sterownik 570 lub nowszy (PyTorch z CUDA 12.8). Przy starszym instalator ostrzega.
- Instalacja ze źródeł (`setup_windows.bat`, venv) działa jak dotąd; macOS bez zmian.

## Zmienione pliki

- **Zmienione:** `main.py`, `worker.py`, `config.py`, `gui/main_window.py`, `huggingface_utils.py`, `requirements.txt`, `setup_windows.ps1`, `README.md`, `.gitignore`, `packaging/LinguaBee-macos.spec` (APP_VERSION 1.1.0).
- **Nowe:**
  - instalator: `packaging/installer/` (`LinguaBee.iss`, `bootstrap.ps1`, `build_installer.ps1`, `compile_locks.ps1`, `downloads.json`, `verify_packages.py`, `Napraw instalacje.bat`);
  - skróty: `packaging/windows/create_shortcut.ps1`, `icons/LinguaBee.ico`;
  - listy pakietów: `requirements/windows-nvidia.txt`, `windows-cpu.txt`, `windows-amd.txt`;
  - pliki startowe: `LinguaBee.bat`, `LinguaBee - konsola.bat`, `Utworz skrot na pulpicie.bat`, `setup_windows.bat`;
  - kod: `torchaudio_compat.py`;
  - pozostałe: `.gitattributes`, `prompts/translation/Tumaczenie_docx.txt`, `RELEASE_NOTES_1.1.0.md`.
- **Usunięte:** `setup.iss`.

## Tag

- Tag: `v1.1.0`

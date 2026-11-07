# LinguaBee

Desktopowa aplikacja do:
- transkrypcji audio/wideo (Whisper),
- tłumaczeń (lokalne i API),
- streszczania (lokalne i API),
- korekty tekstu/SRT,
- eksportu do TXT, DOCX, SRT.

Poniżej znajdziesz **kompletną instrukcję instalacji na Windows i macOS**.

---

## 0) Windows: instalator (zalecane)

Najprostszy sposób instalacji na Windows 10/11 (64-bit), podobny do ComfyUI Desktop: mały
instalator (ok. 20 MB) pobiera resztę z internetu do jednego, wybranego folderu.
Nie trzeba instalować Pythona, Gita ani FFmpeg i nie są potrzebne uprawnienia administratora.

### Instalacja
1. Pobierz `LinguaBee-<wersja>-Setup.exe` (GitHub → Releases) i uruchom.
   Instalator nie jest podpisany cyfrowo, więc Windows może pokazać ostrzeżenie SmartScreen:
   kliknij **Więcej informacji → Uruchom mimo to**.
2. Wybierz folder, np. `C:\LinguaBee` albo `D:\LinguaBee` (najlepiej krótka ścieżka na szybkim
   dysku wewnętrznym z dużą ilością miejsca; nie Program Files i nie OneDrive). Z wolnego dysku
   zewnętrznego (USB) program też działa, ale uruchamia się nawet kilka minut.
3. Wybierz kartę graficzną (instalator podpowiada wykrytą):
   - **NVIDIA (CUDA)** – karty GeForce/RTX, pobieranie ok. 4 GB, na dysku ok. 8,5 GB,
   - **AMD (ROCm)** – Radeon RX 7000/9000, Ryzen AI Max (eksperymentalnie; sterownik Adrenalin 26.2.2+),
   - **Tylko procesor (CPU)** – działa na każdym komputerze (wolniej), pobieranie ok. 1 GB,
     na dysku ok. 4,5 GB.
4. Instalator pobierze Pythona 3.12, PyTorch dla wybranej karty, biblioteki (dokładnie
   przetestowane wersje, sprawdzane sumami SHA-256), FFmpeg i opcjonalnie Deno (YouTube).
   Zależnie od łącza trwa to od kilku do kilkunastu minut.
5. Uruchamiaj program skrótem **LinguaBee** w menu Start lub na pulpicie.

Modele (np. Whisper) pobierają się przy pierwszym użyciu – też do folderu instalacji.

### Co jest w folderze instalacji
| Folder | Zawartość |
|---|---|
| `app\` | kod programu (podmieniany przy aktualizacji) |
| `python\` | Python i biblioteki (w tym PyTorch) |
| `bin\` | `ffmpeg.exe`, `ffprobe.exe`, `uv.exe`, `deno.exe` |
| `models\` | wszystkie modele (Whisper, NLLB, HuggingFace) |
| `output\` | wyniki: TXT, DOCX, SRT, pobrane audio |
| `user\` | ustawienia, prompty, logi (`user\logs`), token HuggingFace |
| `cache\` | pliki tymczasowe bibliotek (można usunąć) |

**Przeniesienie** (także na inny dysk): przenieś cały folder, a potem uruchom instalator
i wskaż nowe miejsce – nic nie jest pobierane ponownie, a skróty i deinstalacja zaczną
wskazywać nowy folder. Bez instalatora pod ręką wystarczy `Napraw instalacje.bat` w nowym
miejscu: poprawi skróty (deinstalację z Ustawień Windows naprawi dopiero instalator).

Modele i wyniki można trzymać gdzie indziej – dopisz do `install.json`
np. `"models_dir": "E:\\Modele"` albo `"output_dir": "D:\\Wyniki"`.

### Naprawa, aktualizacja, deinstalacja
- **Napraw instalację** (menu Start → LinguaBee albo `Napraw instalacje.bat`): sprawdza Pythona,
  biblioteki (także czy ich pliki nie zostały usunięte lub uszkodzone), FFmpeg i Deno i pobiera
  tylko to, czego brakuje. Program musi być wtedy zamknięty. Zmiana karty, np.:
  `"Napraw instalacje.bat" -Gpu cpu` (`auto`, `nvidia`, `amd`, `cpu`).
- **Aktualizacja:** uruchom nowszy instalator i wskaż ten sam folder. Kod zostanie podmieniony,
  biblioteki uzupełnione (pobierane jest tylko to, co się zmieniło), a modele, wyniki
  i ustawienia zostają. Naprawa i aktualizacja instalują też najnowszy yt-dlp (YouTube często
  zmienia sposób udostępniania filmów).
- **Deinstalacja:** Ustawienia → Aplikacje → LinguaBee. Deinstalator zapyta, czy usunąć też
  modele, wyniki i ustawienia (domyślnie zostają).
- Gdy coś nie działa: menu Start → **LinguaBee (konsola)** pokazuje wszystkie komunikaty,
  a przebieg instalacji jest w `user\logs\install.log`.

### Budowanie instalatora (dla autora)
1. `powershell -ExecutionPolicy Bypass -File packaging\installer\compile_locks.ps1` –
   przypięte listy pakietów `requirements\windows-{nvidia,cpu,amd}.txt` z działającego `venv`
   (po zmianie `requirements.txt` albo aktualizacji bibliotek).
2. `powershell -ExecutionPolicy Bypass -File packaging\installer\build_installer.ps1` –
   `dist_installer\LinguaBee-<wersja>-Setup.exe` (wersja z `APP_VERSION` w `config.py`;
   wymaga Inno Setup 6.7+). Przypięte pliki do pobrania (uv, FFmpeg, Deno, CTranslate2 ROCm)
   są w `packaging\installer\downloads.json`.

Instalacja ze źródeł (sekcje 1–8) pozostaje dla osób, które rozwijają program.

---

## 1) Wymagania wstępne

### Obowiązkowe
- Python **3.12.10**
- Git

### Bardzo zalecane
- FFmpeg (do konwersji audio)
- aktualny `yt-dlp`

### Opcjonalne (zależnie od funkcji)
- Ollama (lokalne modele LLM)
- Node.js (pomaga `yt-dlp` przy ekstrakcji YouTube)
- GPU do akceleracji: NVIDIA (CUDA), AMD (ROCm, np. Ryzen AI Max+ / Radeon RDNA2-4) lub Apple Silicon (MPS)

---

## 2) Linki do pobrania

### Python 3.12.10
- Strona wydań Pythona 3.12.10: https://www.python.org/downloads/release/python-31210/
- Windows 64-bit (installer): https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
- macOS universal2 (installer): https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg

### Git
- Strona Git: https://git-scm.com/downloads
- Windows (Git for Windows): https://git-scm.com/download/win
- macOS: https://git-scm.com/download/mac

### FFmpeg
- Oficjalna strona: https://ffmpeg.org/download.html
- Windows buildy (popularne): https://www.gyan.dev/ffmpeg/builds/

### Ollama (opcjonalnie)
- https://ollama.com/download

### Node.js (opcjonalnie, zalecane dla yt-dlp)
- https://nodejs.org/en/download

### Sterowniki NVIDIA + CUDA (opcjonalnie)
- Sterowniki NVIDIA: https://www.nvidia.com/Download/index.aspx
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

### Sterowniki AMD + ROCm (opcjonalnie)
- Sterowniki AMD Adrenalin: https://www.amd.com/en/products/software/adrenalin.html
- Wsparcie ROCm dla PyTorch (Windows i Linux) jest wykrywane i instalowane automatycznie
  przez skrypt `install_pytorch.py` (patrz niżej) - nie trzeba pobierać niczego ręcznie.
- Przy kilku kartach AMD (np. Radeon 8060S + zewnętrzny Radeon AI PRO R9700) skrypt instaluje
  kod dla wszystkich wykrytych kart. PyTorch z ROCm może używać tylko jednej karty w procesie,
  więc LinguaBee pracuje na jednej: kartę wybierasz w ustawieniach Whisper/NLLB/streszczeń
  (zmiana działa po ponownym uruchomieniu) albo zmienną `$env:HIP_VISIBLE_DEVICES="1"`.

### Whisper i NLLB na karcie AMD (CTranslate2 z ROCm, opcjonalnie)
Transkrypcja (faster-whisper) i tłumaczenie NLLB działają na CTranslate2. Wersja z PyPI obsługuje
tylko NVIDIA i CPU, więc na AMD domyślnie liczą na procesorze. Aby przenieść je na kartę AMD:
1. Pobierz `rocm-python-wheels-Windows.zip` (lub `-Linux.zip`) z
   https://github.com/OpenNMT/CTranslate2/releases (testowane: 4.8.4, gfx1151 i gfx1201).
2. Zainstaluj plik pasujący do wersji Pythona, np. dla 3.12:
   `pip install --no-deps --force-reinstall ctranslate2-4.8.4-cp312-cp312-win_amd64.whl`
3. Uruchom LinguaBee - profil AMD sam wykryje, że CTranslate2 widzi kartę, i ustawi na niej
   Whisper oraz NLLB (Whisper w trybie float16).

Uwaga: te paczki korzystają z bibliotek ROCm 7 (`amdhip64_7.dll`) instalowanych razem z PyTorch.
Przy nowszym ROCm sprawdź działanie (`python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"`);
gdy pokaże 0, LinguaBee automatycznie zostanie przy procesorze.

Pomiar (Ryzen AI Max+ 395, large-v3, 5 min nagrania): CPU ok. 260 s, karta AMD float16 ok. 50 s.

---

## 3) Instalacja na Windows (krok po kroku)

### Szybka instalacja skryptem (Windows)
Po zainstalowaniu Pythona 3.12 i Gita (kroki 1–2 niżej) i pobraniu projektu (krok 4)
kliknij dwukrotnie **`setup_windows.bat`** w katalogu projektu. Skrypt:
- tworzy środowisko `.venv` (albo używa istniejącego `venv` / `.venv`),
- instaluje PyTorch dobrany do karty graficznej i pozostałe biblioteki,
- tworzy skrót **LinguaBee** na pulpicie i w menu Start.

To samo z PowerShella: `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1`

Opcjonalnie:
- pomiń tworzenie venv: `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipVenv`
- pomiń instalację pakietów: `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipInstall`
- bez skrótu na pulpicie: `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -NoShortcut`

Potem uruchamiaj program dwuklikiem — patrz sekcja 8.

### Krok 1: Zainstaluj Python 3.12.10
1. Pobierz instalator: https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
2. Uruchom instalator.
3. **Koniecznie zaznacz** opcję `Add Python to PATH`.
4. Kliknij `Install Now`.

Weryfikacja w PowerShell:
- `python --version`
- oczekiwane: `Python 3.12.10`

### Krok 2: Zainstaluj Git
1. Pobierz: https://git-scm.com/download/win
2. Zainstaluj domyślnymi ustawieniami.

Weryfikacja:
- `git --version`

### Krok 3: (Zalecane) Zainstaluj FFmpeg
1. Pobierz build (np. full/shared): https://www.gyan.dev/ffmpeg/builds/
2. Rozpakuj np. do `C:\ffmpeg`.
3. Dodaj `C:\ffmpeg\bin` do zmiennej środowiskowej `PATH`.

Weryfikacja:
- `ffmpeg -version`
- `ffprobe -version`

### Krok 4: Pobierz projekt
W PowerShell:
- `git clone https://github.com/bmalewski/LinguaBee.git`
- `cd LinguaBee`

### Krok 5: Utwórz i aktywuj środowisko wirtualne
- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`

Jeśli PowerShell blokuje skrypty:
- `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Krok 6: Zaktualizuj pip
- `python -m pip install --upgrade pip`

### Krok 7: Zainstaluj PyTorch dopasowany do Twojego sprzętu
- `python install_pytorch.py`

Skrypt wykrywa GPU (NVIDIA/AMD) i instaluje właściwy build PyTorch:
- NVIDIA GPU -> PyTorch CUDA,
- AMD GPU (np. Ryzen AI Max+ 395, Radeon RX 6000/7000/9000) -> PyTorch ROCm,
- brak GPU -> lekki PyTorch CPU-only.

### Krok 8: Zainstaluj pozostałe zależności
- `pip install -r requirements.txt`

### Krok 9: Uruchom aplikację
- `python main.py`
- albo bez PowerShella: dwuklik w `LinguaBee.bat`; skrót z ikoną na pulpicie utworzy
  `Utworz skrot na pulpicie.bat` (patrz sekcja 8)

---

## 4) Instalacja na macOS (krok po kroku)

### Szybka instalacja skryptem (macOS)
W Terminalu, w katalogu projektu:
- `chmod +x ./setup_macos.sh`
- `./setup_macos.sh`

### Krok 1: Zainstaluj Python 3.12.10
1. Pobierz installer: https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg
2. Zainstaluj standardowo.

Weryfikacja w Terminalu:
- `python3 --version`
- oczekiwane: `Python 3.12.10`

### Krok 2: Zainstaluj Git
- Pobierz: https://git-scm.com/download/mac

Weryfikacja:
- `git --version`

### Krok 3: Zainstaluj FFmpeg (zalecane)
Najwygodniej przez Homebrew:
- Homebrew: https://brew.sh/
- potem: `brew install ffmpeg`

Weryfikacja:
- `ffmpeg -version`

### Krok 4: Pobierz projekt
- `git clone https://github.com/bmalewski/LinguaBee.git`
- `cd LinguaBee`

### Krok 5: Utwórz i aktywuj środowisko
- `python3 -m venv .venv`
- `source .venv/bin/activate`

### Krok 6: Zaktualizuj pip
- `python -m pip install --upgrade pip`

### Krok 7: Zainstaluj PyTorch dopasowany do Twojego sprzętu
- `python install_pytorch.py`

Na Apple Silicon skrypt zainstaluje standardowy build PyTorch (obsługa Metal/MPS wbudowana).

### Krok 8: Zainstaluj pozostałe zależności
- `pip install -r requirements.txt`

### Krok 9: Uruchom aplikację
- `python main.py`

---

## 5) Konfiguracja kluczy API

Aplikacja może korzystać z API (Gemini/OpenRouter) przez ustawienia GUI.

**Nie publikuj kluczy API w repozytorium.**
Pliki lokalne z kluczami są ignorowane przez `.gitignore`.

---

## 6) Najczęstsze problemy

### `ffmpeg/ffprobe not found on PATH`
Zainstaluj FFmpeg i upewnij się, że binarka jest w `PATH`.

### Błędy `yt-dlp` przy YouTube (403/cookies)
- zaktualizuj `yt-dlp`: `python -m pip install -U yt-dlp`
- zamknij przeglądarkę przy ekstrakcji cookies,
- doinstaluj Node.js.

### Problemy z GPU/CUDA/ROCm
- sprawdź sterowniki NVIDIA (CUDA) lub AMD Adrenalin (ROCm),
- upewnij się, że uruchomiłeś `python install_pytorch.py` PRZED `pip install -r requirements.txt`
  (w innym wypadku pip może doinstalować domyślny, niewłaściwy build PyTorch),
- sprawdź wykryty profil: `python install_pytorch.py --dry-run`,
- jeśli Twój model GPU AMD nie zostanie automatycznie rozpoznany, zainstaluj ręcznie
  właściwy target: `python install_pytorch.py --force rocm --gfx gfx1151` (pełna lista:
  `python install_pytorch.py --list-targets`),
- dla CPU projekt też działa (wolniej).

## 7) Aktualizacja projektu

W katalogu projektu:
- `git pull`
- `pip install -r requirements.txt --upgrade`

---

## 8) Uruchomienie (skrót)

### Windows: dwuklikiem, bez PowerShella
- **`LinguaBee.bat`** — uruchamia program bez okna konsoli (sam znajduje `venv` / `.venv`).
- **Skrót LinguaBee** na pulpicie i w menu Start (z ikoną) — tworzy go `setup_windows.bat`
  albo **`Utworz skrot na pulpicie.bat`**. Ze skrótu program startuje bez migającej konsoli.
- **`LinguaBee - konsola.bat`** — to samo, ale z oknem konsoli, które zostaje otwarte po
  zamknięciu programu. Użyj go, gdy coś nie działa: widać wszystkie komunikaty i błędy.

Przy starcie bez konsoli komunikaty i błędy trafiają do pliku `logs\linguabee.log` w katalogu
projektu. Jeśli program nie wystartuje, pojawi się okno z opisem błędu i ścieżką do tego pliku.

### Z wiersza poleceń (Windows, macOS)
Po aktywacji `.venv`:
- `python main.py`
- `python main.py --selftest` — sprawdza biblioteki, ffmpeg i kartę graficzną

---

## 9) Budowanie aplikacji .app na macOS (Apple Silicon)

Zamiast uruchamiać program z terminala, można zbudować zwykłą aplikację macOS
(`LinguaBee.app`), uruchamianą dwuklikiem. Build jest przeznaczony na własny komputer
(bez podpisu Apple Developer i notaryzacji), tylko dla Maców z Apple Silicon.

### Wymagania
- Mac z Apple Silicon (M1 lub nowszy), środowisko `venv/` z Pythonem 3.12 i zainstalowanym `requirements.txt`.
- Dostęp do sieci przy pierwszym budowaniu: skrypt instaluje `pyinstaller` oraz pobiera
  statyczne binarki `ffmpeg`/`ffprobe` (arm64, LGPL) z https://ffmpeg.martin-riedl.de/ do `build_assets/bin/`.

### Budowanie
W katalogu projektu:
- `packaging/build_macos.sh` — buduje `dist/LinguaBee.app`
- `packaging/build_macos.sh --install` — dodatkowo kopiuje aplikację do `/Applications`

Pierwsze budowanie trwa kilkanaście minut, a gotowa aplikacja zajmuje ok. 1,1 GB (torch, transformers, Qt).
Modele Whisper/NLLB nie są wbudowane — pobierają się przy pierwszym użyciu, jak dotąd.
Backend tłumaczeń "MLX Apple" nie jest dołączany (mlx-lm koliduje z whisperx — patrz `requirements.txt`).

### Weryfikacja
- `dist/LinguaBee.app/Contents/MacOS/LinguaBee --selftest` — sprawdza, czy wszystkie biblioteki
  i wbudowane `ffmpeg`/`ffprobe` są dostępne wewnątrz aplikacji.
- `dist/LinguaBee.app/Contents/MacOS/LinguaBee` — uruchomienie z terminala pokazuje logi, których
  po dwukliku nie widać.

### Gdzie aplikacja zapisuje dane
Spakowana aplikacja nie zapisuje niczego obok siebie:
- wyniki (TXT/DOCX/SRT, pobrane audio): `~/Documents/LinguaBee`
- modele, ustawienia (`user_settings.json`) i szablony promptów: `~/Library/Application Support/LinguaBee`

Uruchomienie ze źródeł (`python main.py`) nadal używa katalogów `output/`, `models/`, `prompts/` obok `main.py`.

### Instalator .pkg
- `packaging/build_macos.sh --pkg` — buduje aplikację i od razu instalator `dist/LinguaBee-<wersja>.pkg`
- `packaging/build_pkg.sh` — buduje sam instalator z gotowego `dist/LinguaBee.app`

Instalator (ok. 420 MB) kopiuje aplikację do `/Applications` i zastępuje starszą wersję. Wymaga Apple Silicon i macOS 14+;
na Macu z procesorem Intel odmówi instalacji. Ekran powitalny i końcowy instalatora (`packaging/pkg/*.html`) zawierają
instrukcję odblokowania opisaną w sekcji 10.

### Diaryzacja bez bibliotek ffmpeg
Diaryzacja (pyannote 4) domyślnie dekoduje audio przez `torchcodec`, który wymaga bibliotek współdzielonych
ffmpeg (`libav*.dylib`). LinguaBee omija to: dekoduje audio przez PyAV i przekazuje pyannote gotową falę
w pamięci, więc diaryzacja działa zarówno ze źródeł, jak i w zbudowanej aplikacji, bez `brew install ffmpeg`.

---

## 10) Instalacja LinguaBee na innym Macu (instrukcja dla odbiorcy)

LinguaBee nie jest podpisana certyfikatem Apple Developer, dlatego macOS przy pierwszym otwarciu
pliku pobranego lub przeniesionego z innego komputera wyświetla ostrzeżenie:

> **Rzecz „LinguaBee” nie została otwarta.** Apple nie może zweryfikować, czy „LinguaBee” nie zawiera
> szkodliwego oprogramowania, które może uszkodzić Maca lub naruszyć Twoją prywatność.

To zachowanie systemu (Gatekeeper), nie błąd aplikacji. Odblokowanie wykonuje się raz.

### Wymagania
- Mac z procesorem **Apple Silicon** (M1 lub nowszy) — sprawdź w menu Apple → „Ten Mac”.
  Na Macu z procesorem Intel aplikacja nie uruchomi się.
- macOS 14 lub nowszy.

### Krok 1: Uruchom instalator
1. Otwórz plik `LinguaBee-<wersja>.pkg` dwuklikiem.
2. Jeśli pojawi się komunikat, że Apple nie może zweryfikować instalatora, kliknij **Gotowe**
   (nie „Przenieś do Kosza”).
3. Otwórz **Ustawienia systemowe → Prywatność i ochrona**, przewiń do sekcji **Ochrona**.
   Przy wpisie o zablokowaniu instalatora kliknij **Otwórz mimo to** i potwierdź hasłem.
4. Instalator uruchomi się. Przejdź przez kolejne ekrany; aplikacja trafi do katalogu **Programy**.

### Krok 2: Uruchom aplikację
1. Otwórz **LinguaBee** z katalogu Programy (lub z Launchpada).
2. Jeśli ponownie pojawi się ostrzeżenie o braku weryfikacji, powtórz odblokowanie:
   **Gotowe → Ustawienia systemowe → Prywatność i ochrona → Otwórz mimo to**, potem uruchom aplikację jeszcze raz.

Alternatywa dla osób korzystających z Terminala (usuwa znacznik kwarantanny z aplikacji):
- `xattr -dr com.apple.quarantine /Applications/LinguaBee.app`

### Jeśli otrzymałeś samą aplikację (LinguaBee.app), a nie instalator
Skopiuj `LinguaBee.app` do katalogu Programy i wykonaj Krok 2. Jeśli aplikacja przyszła w archiwum ZIP,
rozpakuj je przed skopiowaniem.

### Gdzie aplikacja zapisuje dane
- wyniki (TXT/DOCX/SRT, pobrane audio): `~/Dokumenty/LinguaBee`
- modele, ustawienia i szablony promptów: `~/Biblioteka/Application Support/LinguaBee`

Modele Whisper i inne pobierają się przy pierwszym użyciu (od kilkuset MB do kilku GB), więc potrzebne jest
połączenie z internetem i wolne miejsce na dysku.

### Najczęstsze pytania
- **Po odblokowaniu okno aplikacji pojawia się i znika.** Uruchom aplikację ponownie; przy pierwszym starcie
  nowej wersji zdarza się to jednorazowo.
- **Komunikat o braku ffmpeg.** Nie powinien się pojawić, bo `ffmpeg` jest wbudowany w aplikację.
  Jeśli się pojawi, uruchom w Terminalu `/Applications/LinguaBee.app/Contents/MacOS/LinguaBee --selftest`
  i prześlij wynik autorowi.

# LinguaBee

Desktopowa aplikacja do:
- transkrypcji audio/wideo (Whisper),
- tłumaczeń (lokalne i API),
- streszczania (lokalne i API),
- korekty tekstu/SRT,
- eksportu do TXT, DOCX, HTML, SRT.

Poniżej znajdziesz **kompletną instrukcję instalacji na Windows i macOS**.

---

## 1) Wymagania wstępne

### Obowiązkowe
- Python **3.11.9**
- Git

### Bardzo zalecane
- FFmpeg (do konwersji audio)
- aktualny `yt-dlp`

### Opcjonalne (zależnie od funkcji)
- Ollama (lokalne modele LLM)
- Node.js (pomaga `yt-dlp` przy ekstrakcji YouTube)
- GPU + CUDA (Windows, jeśli chcesz akcelerację)

---

## 2) Linki do pobrania

### Python 3.11.9
- Strona wydań Pythona 3.11.9: https://www.python.org/downloads/release/python-3119/
- Windows 64-bit (installer): https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
- macOS universal2 (installer): https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg

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

---

## 3) Instalacja na Windows (krok po kroku)

### Szybka instalacja skryptem (Windows)
W katalogu projektu uruchom PowerShell i wykonaj:
- `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1`

Opcjonalnie:
- pomiń tworzenie venv: `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipVenv`
- pomiń instalację pakietów: `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipInstall`

### Krok 1: Zainstaluj Python 3.11.9
1. Pobierz instalator: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
2. Uruchom instalator.
3. **Koniecznie zaznacz** opcję `Add Python to PATH`.
4. Kliknij `Install Now`.

Weryfikacja w PowerShell:
- `python --version`
- oczekiwane: `Python 3.11.9`

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

### Krok 7: Zainstaluj zależności
- `pip install -r requirements.txt`

### Krok 8: Uruchom aplikację
- `python main.py`

---

## 4) Instalacja na macOS (krok po kroku)

### Szybka instalacja skryptem (macOS)
W Terminalu, w katalogu projektu:
- `chmod +x ./setup_macos.sh`
- `./setup_macos.sh`

### Krok 1: Zainstaluj Python 3.11.9
1. Pobierz installer: https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg
2. Zainstaluj standardowo.

Weryfikacja w Terminalu:
- `python3 --version`
- oczekiwane: `Python 3.11.9`

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

### Krok 7: Zainstaluj zależności
- `pip install -r requirements.txt`

### Krok 8: Uruchom aplikację
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

### Problemy z GPU/CUDA
- sprawdź sterowniki NVIDIA,
- dla CPU projekt też działa (wolniej).

## 7) Aktualizacja projektu

W katalogu projektu:
- `git pull`
- `pip install -r requirements.txt --upgrade`

---

## 8) Uruchomienie (skrót)

Po aktywacji `.venv`:
- `python main.py`

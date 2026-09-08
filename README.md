# LinguaBee

Desktopowa aplikacja do:
- transkrypcji audio/wideo (Whisper),
- tłumaczeń (lokalne i API),
- streszczania (lokalne i API),
- korekty tekstu/SRT,
- eksportu do TXT, DOCX, SRT.

Poniżej znajdziesz **kompletną instrukcję instalacji na Windows i macOS**.

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
- GPU + CUDA (Windows, jeśli chcesz akcelerację)

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

---

## 3) Instalacja na Windows (krok po kroku)

### Szybka instalacja skryptem (Windows)
W katalogu projektu uruchom PowerShell i wykonaj:
- `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1`

Opcjonalnie:
- pomiń tworzenie venv: `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipVenv`
- pomiń instalację pakietów: `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipInstall`

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

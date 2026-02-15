# Checklist przed wysłaniem projektu na GitHub

## 1) Co WYSYŁAĆ (repo source)
- pliki `.py` z kodem aplikacji
- `README.md`
- `requirements.txt`
- pliki GUI (`gui/`), konfiguracje instalatora (`setup.iss`, `Dockerfile`) jeśli są częścią projektu
- ikony i statyczne zasoby potrzebne do uruchomienia (`icons/`)

## 2) Czego NIE wysyłać (lokalne / wrażliwe)
- kluczy API (`gemini_api_key.txt`, `openai_api_key.txt`, inne sekrety)
- lokalnych danych użytkownika (`user_settings.json`)
- pobranych modeli (`models/`)
- wyników działania programu (`downloads/`, logi)
- katalogów środowiska (`venv/`, `.venv/`)
- cache (`__pycache__/`, `*.pyc`)

## 3) Szybka kontrola przed `git add`
Uruchom i sprawdź, czy nie ma sekretów:
- `git status`
- `git diff -- .gitignore`
- `git grep -n "api_key\|Bearer\|openrouter\|gemini"`

## 4) Pierwszy commit (zalecane)
1. `git init` (jeśli repo jeszcze nie istnieje)
2. `git add .`
3. `git status` i ręczna kontrola listy plików
4. `git commit -m "Initial public version"`

## 5) Połączenie z GitHub
1. utwórz puste repo na GitHub
2. dodaj remote: `git remote add origin <URL_REPO>`
3. push: `git push -u origin main`

## 6) Dodatkowo (bardzo zalecane)
- dodaj `LICENSE`
- dodaj sekcję "Jak skonfigurować klucze API lokalnie" w `README.md`
- rozważ plik `sample_user_settings.json` bez sekretów (tylko przykładowe pola)

# Co commitować, a co zostawić lokalnie

## Commitować
- `*.py`
- `README.md`
- `requirements.txt`
- `Dockerfile`
- `setup.iss`
- `icons/`
- `gui/`
- `prompts/` (jeśli to publiczne, bez danych wrażliwych)

## Nie commitować
- `gemini_api_key.txt`, `openai_api_key.txt`
- `user_settings.json`
- `downloads/`
- `models/`
- `ollama_logs/`
- `venv/`, `.venv/`
- `__pycache__/`, `*.pyc`

## Uwaga praktyczna
Jeśli któryś z plików sekretów był już kiedyś commitowany, samo dodanie do `.gitignore` nie wystarczy.
Trzeba usunąć go z historii lub co najmniej z indeksu Git i zrotować klucze API.

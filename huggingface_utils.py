import os
from pathlib import Path
from huggingface_hub import login, get_token
from huggingface_hub import constants as hf_constants


def _token_path() -> Path:
    """Plik tokenu HuggingFace: $HF_HOME/token (w instalacji: user\\huggingface\\token),
    a gdy HF_HOME nie jest ustawione — domyślnie ~/.cache/huggingface/token."""
    return Path(hf_constants.HF_TOKEN_PATH)


def get_hf_token() -> str | None:
    """
    Pobiera token Hugging Face ze zmiennej środowiskowej lub z pliku.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.getenv(var)
        if token and token.strip():
            return token.strip()

    # Bieżąca lokalizacja (respektuje HF_HOME), potem dawne lokalizacje - tylko do odczytu,
    # żeby token z wcześniejszej instalacji nadal działał.
    candidates = [
        _token_path(),
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    for path in candidates:
        try:
            if path.exists():
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    return token
        except OSError:
            continue

    return None


def save_hf_token(token: str):
    """
    Zapisuje token Hugging Face do pliku i loguje się.
    """
    if not token or not token.strip():
        return
    token = token.strip()

    token_path = _token_path()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token, encoding="utf-8")
    login(token)


def login_huggingface():
    """
    Loguje do Hugging Face, jeśli token jest dostępny.
    """
    token = get_hf_token()
    # Sprawdź, czy już jesteśmy zalogowani, aby uniknąć zbędnych wywołań
    if not token:
        return
    try:
        current = get_token()
    except Exception:
        current = None
    if current != token:
        login(token)

import os
from pathlib import Path
from huggingface_hub import login, HfFolder


def get_hf_token() -> str | None:
    """
    Pobiera token Hugging Face ze zmiennej środowiskowej lub z pliku.
    """
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Sprawdź nową, domyślną lokalizację tokenu
    token_path_cache = Path.home() / ".cache" / "huggingface" / "token"
    if token_path_cache.exists():
        return token_path_cache.read_text(encoding="utf-8").strip()

    # Sprawdź starszą, przestarzałą lokalizację tokenu dla kompatybilności wstecznej
    token_path_legacy = Path.home() / ".huggingface" / "token"
    if token_path_legacy.exists():
        return token_path_legacy.read_text(encoding="utf-8").strip()

    return None


def save_hf_token(token: str):
    """
    Zapisuje token Hugging Face do pliku i loguje się.
    """
    if not token or not token.strip():
        return

    # Zapisz token w nowej, preferowanej lokalizacji
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token, encoding="utf-8")
    login(token)


def login_huggingface():
    """
    Loguje do Hugging Face, jeśli token jest dostępny.
    """
    token = get_hf_token()
    # Sprawdź, czy już jesteśmy zalogowani, aby uniknąć zbędnych wywołań
    if token and HfFolder.get_token() != token:
        login(token)
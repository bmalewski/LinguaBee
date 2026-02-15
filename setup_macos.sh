#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

cd "$ROOT_DIR"

echo "== LinguaBee: setup macOS =="
echo "Katalog: $ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Nie znaleziono python3 w PATH."
  echo "Zainstaluj Python 3.11.9: https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Nie znaleziono git w PATH."
  echo "Zainstaluj Git: https://git-scm.com/download/mac"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Tworzenie środowiska .venv..."
  python3 -m venv .venv
else
  echo "Środowisko .venv już istnieje."
fi

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Brak $VENV_PY"
  exit 1
fi

echo "Aktualizacja pip..."
"$VENV_PY" -m pip install --upgrade pip

echo "Instalacja zależności z requirements.txt..."
"$VENV_PY" -m pip install -r requirements.txt

echo "Aktualizacja yt-dlp (zalecane)..."
"$VENV_PY" -m pip install -U yt-dlp

echo
echo "Setup zakończony."
echo "Uruchomienie aplikacji:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo
echo "Jeśli brakuje ffmpeg: brew install ffmpeg"
echo "Jeśli masz problemy z YouTube, doinstaluj Node.js: https://nodejs.org/en/download"

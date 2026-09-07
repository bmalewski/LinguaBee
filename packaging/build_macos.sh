#!/usr/bin/env bash
# Buduje dist/LinguaBee.app (Apple Silicon, one-dir, podpis ad-hoc).
#
# Użycie:
#   packaging/build_macos.sh            # buduje dist/LinguaBee.app
#   packaging/build_macos.sh --install  # dodatkowo kopiuje do /Applications
#
# Wymagania: Mac z Apple Silicon, venv/ z Pythonem 3.12 (arm64), dostęp do sieci
# przy pierwszym uruchomieniu (pip: pyinstaller, mlx-lm; statyczne ffmpeg/ffprobe).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$(uname -m)" != "arm64" ]; then
  echo "Ten skrypt buduje tylko na Apple Silicon (arm64)." >&2
  exit 1
fi

PY="$ROOT/venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "Brak środowiska: $PY (utwórz venv i zainstaluj requirements.txt)." >&2
  exit 1
fi
"$PY" - <<'EOF'
import platform, sys
assert platform.machine() == "arm64", "venv nie jest arm64"
assert sys.version_info[:2] == (3, 12), f"oczekiwano Pythona 3.12, jest {sys.version.split()[0]}"
EOF

echo "== pip: PyInstaller =="
# Bez -U: aktualizacja zależności podnosiłaby transformers/huggingface-hub do wersji
# niezgodnych z whisperx. Backend MLX (mlx-lm) nie jest dołączany — patrz requirements.txt.
"$PY" -m pip install -q "pyinstaller>=6.14"
if ! "$PY" -m pip check >/dev/null 2>&1; then
  echo "Konflikty zależności w venv (pip check):" >&2
  "$PY" -m pip check >&2 || true
  exit 1
fi

ASSETS="$ROOT/build_assets"
BIN="$ASSETS/bin"
mkdir -p "$BIN"

echo "== ffmpeg / ffprobe (statyczne, arm64) =="
for b in ffmpeg ffprobe; do
  if [ ! -x "$BIN/$b" ]; then
    echo "Pobieram $b z ffmpeg.martin-riedl.de..."
    if curl -fsSL --max-time 300 -o "$BIN/$b.zip" \
         "https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/$b.zip"; then
      unzip -o -q "$BIN/$b.zip" -d "$BIN"
      rm -f "$BIN/$b.zip"
      chmod +x "$BIN/$b"
    else
      cat >&2 <<EOF
Nie udało się pobrać $b. Pobierz ręcznie statyczny build arm64 (LGPL)
z https://ffmpeg.martin-riedl.de/ (macOS -> arm64 -> $b.zip), rozpakuj do:
  $BIN
i nadaj chmod +x. Następnie uruchom skrypt ponownie.
EOF
      exit 1
    fi
  fi
  if ! file "$BIN/$b" | grep -q "Mach-O 64-bit executable arm64"; then
    echo "$BIN/$b nie jest binarką arm64:" >&2; file "$BIN/$b" >&2; exit 1
  fi
  if otool -L "$BIN/$b" | tail -n +2 | grep -vE '/usr/lib/|/System/Library/' | grep -q .; then
    echo "$b ma zależności dylib spoza systemu (nie jest statyczny):" >&2
    otool -L "$BIN/$b" >&2; exit 1
  fi
  echo "  $b: OK ($("$BIN/$b" -version 2>/dev/null | head -1 | cut -d' ' -f1-3))"
done

echo "== ikona .icns =="
ICNS="$ASSETS/LinguaBee.icns"
if [ ! -f "$ICNS" ]; then
  SRC="$ROOT/icons/Gemini_Generated_Image_d5p4z6d5p4z6d5p4.png"   # 1024x1024 RGBA
  SET="$ASSETS/LinguaBee.iconset"
  rm -rf "$SET"; mkdir -p "$SET"
  for s in 16 32 128 256 512; do
    sips -z "$s" "$s" "$SRC" --out "$SET/icon_${s}x${s}.png" >/dev/null
    sips -z "$((s * 2))" "$((s * 2))" "$SRC" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$SET" -o "$ICNS"
  rm -rf "$SET"
  echo "  wygenerowano $ICNS"
else
  echo "  $ICNS już istnieje"
fi

echo "== PyInstaller =="
rm -rf "$ROOT/build" "$ROOT/dist"
"$ROOT/venv/bin/pyinstaller" --noconfirm --clean --log-level WARN "$ROOT/packaging/LinguaBee-macos.spec"

APP="$ROOT/dist/LinguaBee.app"
if [ ! -d "$APP" ]; then
  echo "Budowanie nie utworzyło $APP" >&2
  exit 1
fi
chmod +x "$APP"/Contents/Frameworks/bin/* 2>/dev/null || true

echo "== codesign (ad-hoc) =="
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP" && echo "  podpis OK"

echo "== gotowe =="
du -sh "$APP"
if [ "${1:-}" = "--install" ]; then
  rm -rf /Applications/LinguaBee.app
  cp -R "$APP" /Applications/
  echo "Zainstalowano: /Applications/LinguaBee.app"
fi
echo "Test wbudowanych bibliotek:  \"$APP/Contents/MacOS/LinguaBee\" --selftest"
echo "Uruchomienie z logami:       \"$APP/Contents/MacOS/LinguaBee\""

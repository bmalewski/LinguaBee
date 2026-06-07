"""Izolowany runner przetwarzania audio.

Uruchamiany jako oddzielny proces przez `worker.py`, aby odszumianie /
normalizacja / konwersja do mono działały w izolacji od głównego procesu GUI.
Dzięki temu ewentualne crashe/deadlocki natywnych bibliotek (torch, pyannote,
noisereduce) nie wywracają całej aplikacji.

Protokół komunikacji:
- argv[1]: ścieżka do pliku audio wejściowego
- stdin (bytes, UTF-8): JSON z flagami konfiguracji:
      {"enable_denoising": bool, "enable_normalization": bool, "force_mono": bool}
- stdout (bytes, UTF-8): JSON z wynikiem:
      {"processed_path": "<ścieżka>"}
- kod wyjścia 0 oznacza sukces; każdy inny kod oznacza błąd (worker użyje
  oryginalnego pliku jako fallback).
"""

import sys
import os
import json
from types import SimpleNamespace

# Upewnij się, że katalog projektu (rodzic tego pliku) jest na ścieżce importów.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("Brak ścieżki audio w argumentach.\n")
        return 2

    audio_path = sys.argv[1]
    if not audio_path or not os.path.exists(audio_path):
        sys.stderr.write(f"Plik audio nie istnieje: {audio_path}\n")
        return 2

    # Wczytaj konfigurację z stdin (jeśli dostępna).
    try:
        raw = sys.stdin.buffer.read()
        cfg_dict = json.loads(raw.decode("utf-8")) if raw else {}
    except Exception as e:
        sys.stderr.write(f"Nieprawidłowa konfiguracja na stdin: {e}\n")
        cfg_dict = {}

    config = SimpleNamespace(
        enable_denoising=bool(cfg_dict.get("enable_denoising", False)),
        enable_normalization=bool(cfg_dict.get("enable_normalization", False)),
        force_mono=bool(cfg_dict.get("force_mono", False)),
    )

    # Prosty callback statusu — przekierowuje komunikaty na stderr,
    # aby nie zaśmiecać stdout (zarezerwowany na wynik JSON).
    def status_signal_emit(msg, level="info"):
        sys.stderr.write(f"[{level}] {msg}\n")

    status_signal = SimpleNamespace(emit=status_signal_emit)

    try:
        from audio_processing import process_audio
        processed_path = process_audio(audio_path, config, status_signal)
    except Exception as e:
        sys.stderr.write(f"Błąd przetwarzania audio: {e}\n")
        # Fallback: zwróć oryginalną ścieżkę zamiast całkowicie zawodzić.
        processed_path = audio_path

    sys.stdout.buffer.write(json.dumps({"processed_path": processed_path}).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

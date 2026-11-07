"""Zgodność torchaudio dla bibliotek oczekujących `torchaudio.AudioMetaData`.

Część wersji pyannote, speechbrain i whisperx odwołuje się do `torchaudio.AudioMetaData`,
a nowsze torchaudio trzymają tę klasę tylko w modułach backendu. Funkcję trzeba wywołać przed
pierwszym importem tych bibliotek. Importuje torch (kilka sekund), więc nie należy jej
wywoływać przed pokazaniem okna programu.
"""
import importlib


def ensure_audio_metadata() -> None:
    """Udostępnia `torchaudio.AudioMetaData`, jeśli brak go na najwyższym poziomie pakietu."""
    try:
        import torchaudio
    except Exception:
        return  # brak torchaudio: nic do poprawienia, biblioteki zgłoszą własne błędy
    if hasattr(torchaudio, "AudioMetaData"):
        return
    for module_name in ("torchaudio.backend.common", "torchaudio.backend.sox_io_backend"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "AudioMetaData"):
            torchaudio.AudioMetaData = module.AudioMetaData
            return

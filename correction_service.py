import os

from text_utils import redistribute_text_to_segments
from config import downloads_dir
from model_response_parser import parse_list_response
from artifact_io_service import read_artifact, write_artifact


def _redact_api_key_in_message(msg: str, api_key: str) -> str:
    try:
        if isinstance(api_key, str) and api_key:
            return str(msg).replace(api_key, "REDACTED")
    except Exception:
        pass
    return str(msg)


def run_correction_step(
    config,
    base_name: str,
    file_or_url: str,
    local_path: str,
    text: str,
    segments,
    gemini_rate_limited_until: float,
    status_cb,
    adapters,
):
    corr_mode = getattr(config, "transcription_correction", "Brak")
    corr_prompt = getattr(config, "correction_prompt", "")

    if not corr_mode or corr_mode == "Brak":
        return text, segments, gemini_rate_limited_until

    if not isinstance(corr_prompt, str) or len(corr_prompt.strip()) < 20:
        status_cb("Korekta pominięta: prompt jest pusty lub za krótki (min. 20 znaków).", "warning")
        return text, segments, gemini_rate_limited_until

    status_cb("Rozpoczynam korektę na podstawie zapisanych plików transkryptu...", "info")

    if not adapters.ensure_provider_ready(corr_mode):
        return text, segments, gemini_rate_limited_until

    correction_inputs = []
    for fmt in getattr(config, "formats_original", []):
        ext = str(fmt).lower()
        in_path = os.path.join(downloads_dir, f"{base_name}_original.{ext}")
        if os.path.exists(in_path):
            correction_inputs.append((in_path, ext))

    try:
        if not str(file_or_url).startswith("http") and str(getattr(config, "transcription_model", "")).strip() == "Brak":
            src_ext = os.path.splitext(local_path)[1].lower().lstrip(".")
            if src_ext in {"srt", "docx", "txt"}:
                src_path = local_path
                if os.path.exists(src_path):
                    has_same_ext = any(existing_ext == src_ext for _, existing_ext in correction_inputs)
                    if not has_same_ext:
                        correction_inputs.append((src_path, src_ext))
    except Exception:
        pass

    seen_inputs = set()
    unique_inputs = []
    for in_path, ext in correction_inputs:
        key = os.path.abspath(in_path)
        if key in seen_inputs:
            continue
        seen_inputs.add(key)
        unique_inputs.append((in_path, ext))

    for in_path, ext in unique_inputs:
        try:
            artifact = read_artifact(in_path, ext)
            file_text, file_segments = artifact.text, artifact.segments
        except Exception as e:
            status_cb(f"Korekta: nie udało się odczytać pliku {os.path.basename(in_path)}: {e}", "warning")
            continue

        if not file_text or not file_text.strip():
            status_cb(f"Korekta: pomijam pusty plik {os.path.basename(in_path)}.", "warning")
            continue

        refined = ""
        prompt_for_file = corr_prompt.strip()
        if ext == "srt" and file_segments:
            prompt_for_file += (
                "\n\nINSTRUKCJA: Zwróć poprawione segmenty w postaci JSON-owej listy stringów, "
                "np. [\"seg1\", \"seg2\", ...]. Każdy element listy musi odpowiadać "
                "kolejno segmentowi wejściowemu. NIE dodawaj nic poza czystym JSON-em."
            )
        elif ext in {"txt", "docx"}:
            prompt_for_file += (
                "\n\nINSTRUKCJA TECHNICZNA: To jest zwykły plik tekstowy (TXT/DOCX). "
                "NIE dodawaj znaczników czasowych, numerów segmentów, kodów SRT "
                "ani żadnego formatowania charakterystycznego dla plików .srt. "
                "Podziel tekst na logiczne akapity oddzielone pustą linią — "
                "każdy akapit powinien obejmować jedną spójną myśl lub wątek. "
                "Zwróć wyłącznie poprawiony tekst z akapitami, bez komentarzy i bez markdown."
            )

        refined, gemini_rate_limited_until = adapters.correct_file(
            corr_mode=corr_mode,
            ext=ext,
            file_text=file_text,
            prompt_for_file=prompt_for_file,
            file_segments=file_segments,
            gemini_rate_limited_until=gemini_rate_limited_until,
        )

        if not refined or not refined.strip():
            status_cb(f"Korekta ({ext.upper()}) zwróciła pusty wynik.", "warning")
            continue

        out_path = os.path.join(downloads_dir, f"{base_name}_corrected.{ext}")
        try:
            if ext == "txt":
                write_artifact(text=refined, segments=None, path=out_path, ext=ext)
            elif ext == "docx":
                write_artifact(text=refined, segments=None, path=out_path, ext=ext)
            elif ext == "srt":
                parsed_list = parse_list_response(refined)
                if parsed_list and file_segments:
                    corr_segments = []
                    for i_seg, seg in enumerate(file_segments):
                        txt_val = str(parsed_list[i_seg]).strip() if i_seg < len(parsed_list) else seg.get("text", "")
                        corr_segments.append({"start": seg.get("start", 0), "end": seg.get("end", 0), "text": txt_val})
                    write_artifact(text="", segments=corr_segments, path=out_path, ext=ext)
                    segments = corr_segments
                    text = "\n\n".join([s.get("text", "") for s in corr_segments])
                elif file_segments:
                    sanitized = refined
                    parsed_lines = parse_list_response(refined)
                    if parsed_lines:
                        sanitized = "\n\n".join(parsed_lines)
                    corr_segments = redistribute_text_to_segments(sanitized, file_segments)
                    write_artifact(text="", segments=corr_segments, path=out_path, ext=ext)
                    segments = corr_segments
                    text = "\n\n".join([s.get("text", "") for s in corr_segments])
                else:
                    write_artifact(text=refined, segments=None, path=os.path.join(downloads_dir, f"{base_name}_corrected.txt"), ext="txt")
            status_cb(f"Zapisano korektę: {os.path.basename(out_path)}", "success")
        except Exception as e:
            status_cb(f"Nie udało się zapisać korekty ({ext.upper()}): {e}", "warning")

    return text, segments, gemini_rate_limited_until

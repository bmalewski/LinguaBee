import os
import re
import time

from text_utils import redistribute_text_to_segments
from config import downloads_dir
from model_response_parser import parse_list_response
from artifact_io_service import read_artifact, write_artifact


def _format_eta(eta_seconds: float) -> str:
    if eta_seconds is None:
        return "--:--"
    if eta_seconds < 1:
        return "<1s"
    total = int(max(0, eta_seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02}:{m:02}:{s:02}"
    return f"{m:02}:{s:02}"


def _chunk_srt_segments(segments, max_items: int = 200, max_chars: int = None):
    chunks = []
    current = []
    current_chars = 0
    for seg in segments or []:
        txt = str(seg.get("text", "")).strip()
        add_len = len(txt) + 8
        if current and (len(current) >= max_items or (max_chars is not None and (current_chars + add_len) > max_chars)):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(seg)
        current_chars += add_len
    if current:
        chunks.append(current)
    return chunks


def _split_text_units_for_correction(text: str):
    src = str(text or "").strip()
    if not src:
        return []

    units = [p.strip() for p in re.split(r"\n\s*\n", src) if p and p.strip()]
    if len(units) <= 1:
        units = [ln.strip() for ln in src.splitlines() if ln and ln.strip()]

    if len(units) <= 1 and len(src) > 4000:
        sentence_parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", src) if s and s.strip()]
        if len(sentence_parts) > 1:
            units = sentence_parts

    return units if units else [src]


def _chunk_text_units(units, max_items: int = 200, max_chars: int = 14000):
    chunks = []
    current = []
    current_chars = 0
    safe_max_items = max(1, int(max_items or 200))
    safe_max_chars = max(2000, int(max_chars or 14000))

    for unit in units or []:
        txt = str(unit).strip()
        if not txt:
            continue
        add_len = len(txt) + 2
        if current and (len(current) >= safe_max_items or (current_chars + add_len) > safe_max_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(txt)
        current_chars += add_len

    if current:
        chunks.append(current)
    return chunks


def _correct_text_in_batched_chunks(
    file_text: str,
    base_prompt: str,
    provider_name: str,
    call_provider,
    batch_size: int = 200,
    status_cb=None,
    progress_cb=None,
):
    units = _split_text_units_for_correction(file_text)
    if not units:
        return ""

    chunks = _chunk_text_units(units, max_items=batch_size)
    if len(chunks) <= 1:
        return call_provider(file_text, base_prompt)

    corrected_chunks = []
    processed_units = 0
    total_units = len(units)
    started_at = time.time()

    try:
        if progress_cb:
            progress_cb(0)
    except Exception:
        pass

    for idx, chunk_units in enumerate(chunks):
        chunk_text = "\n\n".join(chunk_units)
        prompt_for_chunk = (
            str(base_prompt or "").strip()
            + "\n\nINSTRUKCJA TECHNICZNA: To jest część "
            + f"{idx + 1}/{len(chunks)} długiego dokumentu. "
            + "Popraw tylko treść tej części, bez komentarzy i bez markdown."
        )

        try:
            if status_cb:
                status_cb(
                    f"Korekta {provider_name} TXT/DOCX: paczka {idx + 1}/{len(chunks)} (jednostek: {len(chunk_units)})",
                    "info",
                )
        except Exception:
            pass

        refined_chunk = str(call_provider(chunk_text, prompt_for_chunk) or "").strip()
        if not refined_chunk:
            refined_chunk = chunk_text
        corrected_chunks.append(refined_chunk)

        processed_units += len(chunk_units)
        try:
            if progress_cb and total_units > 0:
                progress_cb(int((processed_units / total_units) * 100))
        except Exception:
            pass

        try:
            if status_cb and total_units > 0:
                elapsed = max(0.001, time.time() - started_at)
                rate = processed_units / elapsed
                remaining = max(0, total_units - processed_units)
                eta_seconds = (remaining / rate) if rate > 0 else None
                status_cb(
                    f"Korekta {provider_name} TXT/DOCX postęp: {processed_units}/{total_units} | ETA: {_format_eta(eta_seconds)}",
                    "info",
                )
        except Exception:
            pass

    try:
        if progress_cb:
            progress_cb(100)
    except Exception:
        pass

    return "\n\n".join(part for part in corrected_chunks if str(part).strip()).strip()


def _correct_srt_with_batched_provider(
    provider_name: str,
    prompt: str,
    file_segments: list,
    send_batch,
    batch_size: int = 200,
    status_cb=None,
    progress_cb=None,
    inter_batch_sleep_s: float = 0.0,
):
    if not file_segments:
        return []

    chunks = _chunk_srt_segments(file_segments, max_items=max(1, int(batch_size or 200)))
    try:
        if progress_cb:
            progress_cb(0)
    except Exception:
        pass

    corrected_lines = []
    started_at = time.time()
    total_segments = len(file_segments)
    for idx, chunk in enumerate(chunks):
        try:
            if status_cb:
                status_cb(
                    f"Korekta {provider_name} SRT: paczka {idx + 1}/{len(chunks)} (segmentów: {len(chunk)})",
                    "info",
                )
        except Exception:
            pass

        numbered = []
        for i, seg in enumerate(chunk, start=1):
            numbered.append(f"{i}. {str(seg.get('text', '')).strip()}")

        batch_prompt = (
            prompt.strip()
            + "\n\nINSTRUKCJA: Otrzymasz numerowaną listę segmentów SRT."
            + " Zwróć WYŁĄCZNIE JSON-ową listę stringów, bez komentarzy i bez markdown."
            + " Każdy element listy musi odpowiadać jednemu wejściowemu segmentowi, w tej samej kolejności."
            + f"\nTo jest paczka {idx + 1}/{len(chunks)}."
        )

        response = send_batch(batch_prompt, "\n".join(numbered))
        parsed = parse_list_response(response)
        if not parsed:
            return None

        for i, seg in enumerate(chunk):
            if i < len(parsed) and str(parsed[i]).strip():
                corrected_lines.append(str(parsed[i]).strip())
            else:
                corrected_lines.append(str(seg.get("text", "")).strip())

        try:
            if progress_cb:
                pct = int(((idx + 1) / max(1, len(chunks))) * 100)
                progress_cb(max(0, min(100, pct)))
        except Exception:
            pass

        try:
            if status_cb and total_segments > 0:
                processed = min(len(corrected_lines), total_segments)
                elapsed = max(0.001, time.time() - started_at)
                rate = processed / elapsed
                remaining = max(0, total_segments - processed)
                eta_seconds = (remaining / rate) if rate > 0 else None
                status_cb(f"Korekta {provider_name} SRT postęp: {processed}/{total_segments} segmentów | ETA: {_format_eta(eta_seconds)}", "info")
        except Exception:
            pass

        if inter_batch_sleep_s > 0 and idx < len(chunks) - 1:
            time.sleep(inter_batch_sleep_s)

    if len(corrected_lines) < len(file_segments):
        corrected_lines.extend([str(s.get("text", "")).strip() for s in file_segments[len(corrected_lines):]])
    return corrected_lines[:len(file_segments)]


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

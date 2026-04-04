from docx import Document
import os
import av
import re
import textwrap
from config import downloads_dir

def format_timestamp(seconds):
    """Converts seconds into a SRT timestamp format (HH:MM:SS,ms)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds * 1000) % 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def save_txt(text, path):
    """Saves plain text to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_docx(text, path):
    """Saves text to a DOCX file."""
    doc = Document()
    # Split text by double newlines to create paragraphs
    for para in text.split('\n\n'):
        if para.strip():
            doc.add_paragraph(para.strip())
    doc.save(path)

def _split_text_for_srt(text: str, max_chars_per_line: int, max_lines: int):
    raw = (text or "").strip()
    if not raw:
        return []

    if max_chars_per_line <= 0:
        return [raw]

    wrapped = textwrap.wrap(
        raw,
        width=max_chars_per_line,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        wrapped = [raw]

    if max_lines <= 0:
        return ["\n".join(wrapped)]

    blocks = []
    for i in range(0, len(wrapped), max_lines):
        chunk = wrapped[i:i + max_lines]
        if chunk:
            blocks.append("\n".join(chunk))
    return blocks


def save_srt(segments, path, max_lines=None, max_chars_per_line=None):
    """Saves transcription segments to an SRT subtitle file."""
    if not segments:
        return

    try:
        max_lines = int(max_lines if max_lines is not None else 2)
    except Exception:
        max_lines = 2
    try:
        max_chars_per_line = int(max_chars_per_line if max_chars_per_line is not None else 25)
    except Exception:
        max_chars_per_line = 25

    with open(path, "w", encoding="utf-8") as f:
        cue_idx = 1
        for seg in segments:
            start = format_timestamp(seg.get("start", 0))
            end = format_timestamp(seg.get("end", seg.get("start", 0)))
            text = seg.get("text", "").strip()
            if not text:
                continue

            blocks = _split_text_for_srt(text, max_chars_per_line=max_chars_per_line, max_lines=max_lines)
            if not blocks:
                continue

            if len(blocks) == 1:
                f.write(f"{cue_idx}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{blocks[0]}\n\n")
                cue_idx += 1
                continue

            seg_start = float(seg.get("start", 0.0) or 0.0)
            seg_end = float(seg.get("end", seg_start) or seg_start)
            duration = max(0.001, seg_end - seg_start)
            block_duration = duration / len(blocks)

            for block_i, block_text in enumerate(blocks):
                block_start = seg_start + (block_i * block_duration)
                block_end = seg_end if block_i == len(blocks) - 1 else block_start + block_duration
                f.write(f"{cue_idx}\n")
                f.write(f"{format_timestamp(block_start)} --> {format_timestamp(block_end)}\n")
                f.write(f"{block_text}\n\n")
                cue_idx += 1


def _parse_srt_timestamp(value: str) -> float:
    """Parses SRT timestamp HH:MM:SS,mmm into seconds."""
    m = re.match(r"\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*", value or "")
    if not m:
        raise ValueError(f"Nieprawidłowy znacznik czasu SRT: {value}")
    h, mm, s, ms = m.groups()
    return int(h) * 3600 + int(mm) * 60 + int(s) + int(ms) / 1000.0


def load_srt(path):
    """Loads SRT file and returns (full_text, segments)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    # Split blocks by blank lines (supports CRLF/LF)
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip(), flags=re.MULTILINE)
    segments = []

    for block in blocks:
        lines = [ln.rstrip("\r") for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue

        # Optional numeric index at first line
        if re.fullmatch(r"\d+", lines[0].strip()):
            lines = lines[1:]
        if not lines:
            continue

        # Time line: "start --> end"
        time_line = lines[0]
        if "-->" not in time_line:
            continue

        start_raw, end_raw = [x.strip() for x in time_line.split("-->", 1)]
        try:
            start = _parse_srt_timestamp(start_raw)
            end = _parse_srt_timestamp(end_raw)
        except Exception:
            continue

        text = "\n".join(lines[1:]).strip()
        segments.append({"start": start, "end": end, "text": text})

    full_text = "\n\n".join(seg.get("text", "") for seg in segments if seg.get("text", "").strip())
    return full_text, segments

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm")

def is_video_file(file_path):
    """Checks if a file is a video based on its extension."""
    return file_path.lower().endswith(VIDEO_EXTENSIONS)

def extract_audio_from_video(video_path, status_signal=None, progress_signal=None):
    """
    Extracts audio from a video file, saves it as an MP3 file in the downloads directory,
    and reports progress.
    Returns the path to the extracted audio file.
    """
    if status_signal:
        status_signal.emit(f"Wykryto plik wideo. Rozpoczynanie ekstrakcji audio z: {os.path.basename(video_path)}", "info")

    try:
        output_audio_path = os.path.join(
            downloads_dir,
            f"{os.path.splitext(os.path.basename(video_path))[0]}_audio.mp3"
        )

        with av.open(video_path) as input_container:
            if not input_container.streams.audio:
                if status_signal:
                    status_signal.emit(f"Błąd: Plik wideo nie zawiera ścieżki audio: {os.path.basename(video_path)}", "error")
                return None

            input_stream = input_container.streams.audio[0]
            total_duration_sec = input_stream.duration * input_stream.time_base if input_stream.duration else None

            if progress_signal:
                progress_signal.emit(0)

            with av.open(output_audio_path, 'w') as output_container:
                output_stream = output_container.add_stream('mp3')

                for frame in input_container.decode(input_stream):
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)
                    
                    if progress_signal and total_duration_sec and total_duration_sec > 0:
                        current_time_sec = frame.pts * input_stream.time_base
                        progress = int((current_time_sec / total_duration_sec) * 100)
                        progress_signal.emit(progress)

                # Flush any remaining frames
                for packet in output_stream.encode(None):
                    output_container.mux(packet)

        if progress_signal:
            progress_signal.emit(100)
        if status_signal:
            status_signal.emit(f"Ekstrakcja audio zakończona. Plik zapisano w: {output_audio_path}", "info")
        
        return output_audio_path

    except av.AVError as e:
        if status_signal:
            status_signal.emit(f"Błąd PyAV podczas ekstrakcji audio: {e}", "error")
        return None
    except Exception as e:
        if status_signal:
            status_signal.emit(f"Nieoczekiwany błąd podczas ekstrakcji audio: {e}", "error")
        return None

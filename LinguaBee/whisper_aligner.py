"""
Lightweight optional forced-alignment helper using whisperx when available.
If `whisperx` is not installed, functions here are no-ops and return the original
segments unchanged.

This module exposes `forced_align_refined_text(refined_text, audio_path, segments, model)`
which will try to run whisperx forced_alignment to map words from `refined_text`
back to timestamps. It returns a new `segments` list suitable for SRT output.

Note: whisperx has its own model download/runtime requirements. The helper
catches ImportError and other exceptions and logs via a `logger`-like callable
if provided.
"""
from typing import List, Dict, Optional
import os


def forced_align_refined_text(refined_text: str, audio_path: str, segments: List[Dict], model: Optional[str] = None, status_cb=None) -> List[Dict]:
    """Attempt to force-align refined_text to the audio and produce updated segments.

    - refined_text: corrected transcript (string)
    - audio_path: path to the audio file used for the original transcription
    - segments: original segments (list of dicts with 'start','end','text')
    - model: optional whisperx model name (e.g., 'large-v2')
    - status_cb: optional callable(status_message, level) for UI logging

    Returns: new_segments (list)
    """
    try:
        import whisperx
    except Exception as e:
        if status_cb:
            try:
                status_cb(f"Forced-alignment skipped: whisperx not available ({e})", "warning")
            except Exception:
                pass
        return segments

    # Try to run whisperx forced alignment.
    try:
        if status_cb:
            try:
                status_cb("Rozpoczynam forced-alignment (whisperx)...", "info")
            except Exception:
                pass

        # Load whisper model (delegates to whisperx's utilities)
        device = 'cuda' if hasattr(whisperx, 'torch') and whisperx.torch.cuda.is_available() else 'cpu'
        model_name = model or 'large-v2'

        # Run whisperx inference (speech-to-text) with word/timestamp info then align
        # NOTE: We run a short pipeline: transcribe audio to get alignment object, then replace text.
        # whisperx requires the original model and a language model for alignment; this is a heuristic
        # usage: whisperx imposes extra downloads; we try to reuse the provided segments' timestamps.

        # transcribe with whisperx (this may re-run transcription; if expensive, consider optimizing)
        audio = audio_path
        asr_model = whisperx.load_model(model_name, device)
        result = asr_model.transcribe(audio)

        # The result contains words with timestamps; perform forced alignment using provided corrected text
        # whisperx provides a forced_alignment function
        alignment = whisperx.align(result['segments'], asr_model, audio, device)

        # Build new segments by mapping words from refined_text to timestamps in `alignment`.
        # For now, use a conservative approach: split refined_text into words and assign them in order
        # to the aligned words timestamps, then rebuild segments by grouping contiguous words into
        # buckets that approximate original segments durations.

        aligned_words = alignment.get('words', []) if isinstance(alignment, dict) else []
        if not aligned_words:
            # alignment didn't produce words; fallback
            if status_cb:
                try:
                    status_cb("Forced-alignment nie zwrócił zaanotowanych słów; używam heurystyki.", "warning")
                except Exception:
                    pass
            return segments

        # Split refined_text into words
        refined_words = refined_text.split()
        if not refined_words:
            return segments

        # Map refined words to aligned_words one-to-one as far as possible
        mapped = []
        for i, rw in enumerate(refined_words):
            if i < len(aligned_words):
                aw = aligned_words[i]
                mapped.append({'word': rw, 'start': aw.get('start'), 'end': aw.get('end')})
            else:
                # no timestamp for this word, assign to last known timestamp
                last = aligned_words[-1]
                mapped.append({'word': rw, 'start': last.get('end'), 'end': last.get('end')})

        # Now rebuild segments: create buckets corresponding to original segments count
        new_segments = []
        if not segments:
            # create a single segment covering entire duration
            total_start = mapped[0]['start'] if mapped else 0.0
            total_end = mapped[-1]['end'] if mapped else 0.0
            new_segments.append({'start': total_start, 'end': total_end, 'text': ' '.join([m['word'] for m in mapped])})
            return new_segments

        # Determine target word counts proportional to segment durations
        seg_durations = []
        total_dur = 0.0
        for seg in segments:
            s = seg.get('start', 0.0)
            e = seg.get('end', s)
            d = max(0.0, e - s)
            seg_durations.append(d)
            total_dur += d
        if total_dur <= 0:
            # evenly distribute words
            per_seg = max(1, len(mapped) // len(segments))
            idx = 0
            for seg in segments:
                part = mapped[idx:idx+per_seg]
                idx += per_seg
                if not part:
                    text = ''
                    s = seg.get('start', 0.0)
                    e = seg.get('end', s)
                else:
                    text = ' '.join([p['word'] for p in part])
                    s = part[0]['start'] or seg.get('start', 0.0)
                    e = part[-1]['end'] or seg.get('end', s)
                new_segments.append({'start': s, 'end': e, 'text': text})
            return new_segments

        # Assign words proportional to duration
        words_total = len(mapped)
        assigned = 0
        cursor = 0
        for i, seg in enumerate(segments):
            proportion = seg_durations[i] / total_dur if total_dur > 0 else 1.0 / len(segments)
            take = int(round(proportion * words_total))
            if i == len(segments) - 1:
                take = words_total - assigned
            part = mapped[cursor:cursor+take]
            cursor += take
            assigned += take
            if not part:
                text = ''
                s = seg.get('start', 0.0)
                e = seg.get('end', s)
            else:
                text = ' '.join([p['word'] for p in part])
                s = part[0]['start'] or seg.get('start', 0.0)
                e = part[-1]['end'] or seg.get('end', s)
            new_segments.append({'start': s, 'end': e, 'text': text})

        if status_cb:
            try:
                status_cb("Forced-alignment zakończony.", "info")
            except Exception:
                pass

        return new_segments

    except Exception as e:
        if status_cb:
            try:
                status_cb(f"Forced-alignment nie powiódł się: {e}", "warning")
            except Exception:
                pass
        return segments
*** End Patch
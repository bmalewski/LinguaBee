import re

def chunk_text(text, max_chunk_size=400):
    """Splits text into chunks of max_chunk_size, trying to break at sentence or paragraph boundaries."""
    if not text:
        return []

    # Try to split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_chunk_len = 0

    for para in paragraphs:
        # If adding the next paragraph exceeds max_chunk_size, start a new chunk
        if current_chunk_len + len(para) + 2 > max_chunk_size and current_chunk: # +2 for potential \n\n
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_chunk_len = 0
        
        current_chunk.append(para)
        current_chunk_len += len(para) + 2 # +2 for \n\n
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # If any chunk is still too large (e.g., a single very long paragraph), split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', chunk) # Split by sentences
            sub_chunk = []
            sub_chunk_len = 0
            for sent in sentences:
                if sub_chunk_len + len(sent) + 1 > max_chunk_size and sub_chunk: # +1 for space

                    final_chunks.append(" ".join(sub_chunk))
                    sub_chunk = []
                    sub_chunk_len = 0
                sub_chunk.append(sent)
                sub_chunk_len += len(sent) + 1
            if sub_chunk:
                final_chunks.append(" ".join(sub_chunk))
        else:
            final_chunks.append(chunk)
            
    return final_chunks

def add_missing_spaces(text):
    if not text:
        return ""
    # Regex to find a period, question mark, or exclamation mark followed by a letter or opening parenthesis, without a space in between
    # Excludes cases like "..." or "1.23"
    # Also handles cases like "word. (next sentence)"
    corrected_text = re.sub(r'([.!?])([A-Za-z(])', r'\1 \2', text)
    return corrected_text


def format_transcript(title: str, text: str, segments: list = None) -> str:
    """Return a formatted transcript string with a title and paragraph breaks.

    - title: string to place at top (e.g., recording title)
    - text: raw transcript text
    - segments: optional list of segment dicts with 'start' and 'end' (seconds) and 'text'
    Heuristics:
    - If segments provided, create paragraphs by grouping segments where gap between segments > 2.0s
    - Otherwise, split into paragraphs by double-newline or by grouping sentences into ~200-400 char paragraphs
    """
    if not text:
        return title + "\n\n"

    paragraphs = []
    if segments:
        # segments expected to be list of dicts with 'start' and 'text' keys (whisper-like)
        # We'll accumulate segment texts into paragraphs but avoid breaking inside sentences.
        current_para = []
        current_para_text = ''
        current_para_len = 0
        last_end = None
        for seg in segments:
            seg_text = (seg.get('text') or '').strip()
            start = seg.get('start')
            end = seg.get('end')

            # If there's a large time gap between segments, force paragraph boundary
            large_gap = last_end is not None and start is not None and (start - last_end) > 2.0

            if seg_text:
                if current_para_text:
                    # join with a space to avoid accidental concatenation
                    current_para_text += ' ' + seg_text
                else:
                    current_para_text = seg_text
                current_para_len = len(current_para_text)

            # Decide whether to end the paragraph:
            # - large gap
            # - current paragraph ends with terminal punctuation
            # - or paragraph length exceeds threshold (fallback)
            ends_with_punct = bool(current_para_text and current_para_text.strip()[-1] in '.!?')
            if seg_text and (large_gap or ends_with_punct or current_para_len > 600):
                # finalize paragraph
                paragraphs.append(current_para_text.strip())
                current_para_text = ''
                current_para_len = 0

            last_end = end or start or last_end

        # Flush any remaining text as a paragraph
        if current_para_text:
            paragraphs.append(current_para_text.strip())
    else:
        # No segments: try to split by existing blank lines first
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paras and len(paras) > 1:
            paragraphs = paras
        else:
            # Group sentences into paragraphs of ~300 chars
            sentences = re.split(r'(?<=[.!?])\s+', add_missing_spaces(text))
            cur = []
            cur_len = 0
            for s in sentences:
                if cur_len + len(s) + 1 > 300 and cur:
                    paragraphs.append(' '.join(cur).strip())
                    cur = []
                    cur_len = 0
                cur.append(s)
                cur_len += len(s) + 1
            if cur:
                paragraphs.append(' '.join(cur).strip())

    # Build final string: title (as a styled heading) then paragraphs separated by double newlines
    lines = []
    if title:
        # Apply simple title styling: capitalize and add underlining line
        t = title.strip()
        title_line = t.title()
        underline = '-' * len(title_line)
        lines.append(title_line)
        lines.append(underline)
        lines.append('')
    for p in paragraphs:
        # Avoid splitting paragraphs mid-sentence: ensure paragraph ends with terminal punctuation
        p = p.strip()
        if p and p[-1] not in '.!?':
            # If paragraph doesn't end with punctuation, try to find a sensible sentence boundary near the end
            # If none, keep as-is (better to not invent punctuation)
            pass
        lines.append(p)
        lines.append('')

    return '\n'.join(lines).strip() + '\n'


def redistribute_text_to_segments(text: str, segments: list) -> list:
    """Distribute a corrected transcript `text` across `segments` returning
    an updated segments list where each segment's 'text' is replaced with the
    corresponding portion. This is a simple heuristic that splits on word
    boundaries trying to keep character counts roughly balanced per segment.

    This is intentionally conservative: it preserves ordering and doesn't
    attempt forced-alignment. For better accuracy consider using a word-level
    forced-alignment tool (e.g., `whisperx` or Gentle) to map corrected words
    back to timestamps.
    """
    if not text or not segments:
        return segments

    words = text.split()
    if not words:
        return segments

    total_chars = len(text)
    n = len(segments)
    # target characters per segment (at least 1)
    target = max(1, total_chars // n)

    seg_texts = []
    cur_words = []
    cur_len = 0
    for w in words:
        cur_words.append(w)
        cur_len += len(w) + 1
        # finalize current bucket when we've reached target for all but last segment
        if cur_len >= target and len(seg_texts) < n - 1:
            seg_texts.append(" ".join(cur_words).strip())
            cur_words = []
            cur_len = 0

    # remaining words -> last segment (may be empty)
    seg_texts.append(" ".join(cur_words).strip())

    # If we somehow ended with fewer parts (rare), pad with empty strings
    while len(seg_texts) < n:
        seg_texts.append("")

    # If we have more parts than segments, merge extras into the last segment
    if len(seg_texts) > n:
        seg_texts = seg_texts[:n-1] + [" ".join(seg_texts[n-1:]).strip()]

    # Assign back to segments (preserve original start/end)
    for i, seg in enumerate(segments):
        try:
            seg_text = seg_texts[i] if i < len(seg_texts) else seg.get('text', '')
            seg['text'] = seg_text
        except Exception:
            # if assignment fails for some segment, leave it unchanged
            pass

    return segments

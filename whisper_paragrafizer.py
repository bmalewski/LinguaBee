import re
from typing import List, Dict, Any, Optional

# whisper_paragrafizer.py
# Small, dependency-free paragraphizer for transcription segments.
# Strategy: build sentences from word-level timestamps (if available),
# then group sentences into paragraphs using silence thresholds, max char
# limits and optional speaker changes.

PUNCT_RE = re.compile(r"[\.\!?…]\s*$")


def _safe_get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def paragraphize_segments(
    segments: List[Dict[str, Any]],
    *,
    silence_threshold: float = 1.0,
    max_chars: int = 500,
    break_on_speaker: bool = True,
    min_sentence_chars: int = 10,
) -> List[Dict[str, Any]]:
    """Convert aligned `segments` into paragraphs.

    segments: list of dicts produced by transcription/alignment. Expected keys:
      - 'start' / 'end' (or 'begin'/'finish')
      - 'text'
      - optional: 'speaker'
      - optional: 'words' -> list of {'start','end','word'}

    Returns list of paragraphs. Paragraph dict fields:
      - 'text', 'start', 'end', 'speakers' (set), 'sentences' (list)

    This function is intentionally heuristic and dependency-free so it can be
    imported anywhere in the project. It tries to preserve timestamps when
    word-level timestamps exist.
    """

    # Build a normalized word list (timestamped tokens)
    words = []  # list of dicts: {text,start,end,speaker}
    for seg in segments:
        seg_start = _safe_get(seg, 'start', 'begin', default=0.0) or 0.0
        seg_end = _safe_get(seg, 'end', 'finish', default=seg_start) or seg_start
        seg_speaker = seg.get('speaker')
        if isinstance(seg.get('words'), list) and seg.get('words'):
            for w in seg['words']:
                w_text = w.get('word') if isinstance(w, dict) else str(w)
                w_start = w.get('start', seg_start) if isinstance(w, dict) else seg_start
                w_end = w.get('end', seg_end) if isinstance(w, dict) else seg_end
                words.append({'text': w_text.strip(), 'start': float(w_start), 'end': float(w_end), 'speaker': seg_speaker})
        else:
            # Fallback: treat whole segment as a single token
            t = seg.get('text') or ''
            words.append({'text': t.strip(), 'start': float(seg_start), 'end': float(seg_end), 'speaker': seg_speaker})

    # If there are no words, return a single paragraph containing joined text
    if not words:
        joined = "\n".join(s.get('text', '') for s in segments)
        start = min((s.get('start', s.get('begin', 0.0)) or 0.0) for s in segments) if segments else 0.0
        end = max((s.get('end', s.get('finish', 0.0)) or 0.0) for s in segments) if segments else 0.0
        return [{'text': joined, 'start': start, 'end': end, 'speakers': set(), 'sentences': []}]

    # Build sentences from the words
    sentences = []
    cur_words = []
    for i, w in enumerate(words):
        cur_words.append(w)
        is_last = (i == len(words) - 1)

        # compute gap to next word
        gap = 0.0
        if not is_last:
            gap = max(0.0, words[i+1]['start'] - w['end'])

        ends_with_punct = bool(PUNCT_RE.search(w['text']))
        too_short = sum(len(x['text']) for x in cur_words) < min_sentence_chars

        # close sentence on punctuation or large gap or if next is dramatic silence
        if ends_with_punct or gap > silence_threshold or is_last:
            sent_text = " ".join(x['text'] for x in cur_words).strip()
            s_start = cur_words[0]['start']
            s_end = cur_words[-1]['end']
            # most common speaker in the sentence (if available)
            speakers = [x['speaker'] for x in cur_words if x.get('speaker') is not None]
            speaker = None
            if speakers:
                try:
                    # majority vote
                    speaker = max(set(speakers), key=speakers.count)
                except Exception:
                    speaker = speakers[0]

            sentences.append({'text': sent_text, 'start': s_start, 'end': s_end, 'speaker': speaker})
            cur_words = []

    # Group sentences into paragraphs
    paragraphs = []
    cur_para = {'sentences': [], 'text_chars': 0, 'speakers': set()}
    for idx, s in enumerate(sentences):
        if not cur_para['sentences']:
            # start new paragraph
            cur_para['sentences'].append(s)
            cur_para['text_chars'] += len(s['text'])
            if s.get('speaker') is not None:
                cur_para['speakers'].add(s['speaker'])
            continue

        prev = cur_para['sentences'][-1]
        gap = max(0.0, s['start'] - prev['end'])
        speaker_change = (s.get('speaker') is not None and prev.get('speaker') is not None and s.get('speaker') != prev.get('speaker'))

        should_break = False
        if break_on_speaker and speaker_change and gap > 0.15:
            # small tolerance so short overlaps don't break
            should_break = True
        if gap > silence_threshold:
            should_break = True
        if cur_para['text_chars'] + len(s['text']) > max_chars:
            should_break = True

        if should_break:
            # finalize current paragraph
            para_text = " ".join(sent['text'] for sent in cur_para['sentences']).strip()
            para_start = cur_para['sentences'][0]['start']
            para_end = cur_para['sentences'][-1]['end']
            paragraphs.append({'text': para_text, 'start': para_start, 'end': para_end, 'speakers': set(cur_para['speakers']), 'sentences': cur_para['sentences']})
            # start new
            cur_para = {'sentences': [s], 'text_chars': len(s['text']), 'speakers': set()}
            if s.get('speaker') is not None:
                cur_para['speakers'].add(s['speaker'])
        else:
            cur_para['sentences'].append(s)
            cur_para['text_chars'] += len(s['text'])
            if s.get('speaker') is not None:
                cur_para['speakers'].add(s['speaker'])

    # flush remaining paragraph
    if cur_para['sentences']:
        para_text = " ".join(sent['text'] for sent in cur_para['sentences']).strip()
        para_start = cur_para['sentences'][0]['start']
        para_end = cur_para['sentences'][-1]['end']
        paragraphs.append({'text': para_text, 'start': para_start, 'end': para_end, 'speakers': set(cur_para['speakers']), 'sentences': cur_para['sentences']})

    return paragraphs


def paragraphs_to_plaintext(paragraphs: List[Dict[str, Any]]) -> str:
    """Convert paragraph list to a human-friendly plaintext (double newline between paragraphs)."""
    out = []
    for p in paragraphs:
        out.append(p.get('text', '').strip())
    return "\n\n".join(out)


if __name__ == '__main__':
    # quick self-test
    sample_segments = [
        {'start': 0.0, 'end': 2.0, 'text': 'Cześć wszystkim', 'words': [{'word':'Cześć','start':0.0,'end':0.4},{'word':'wszystkim','start':0.4,'end':2.0}]},
        {'start': 3.5, 'end': 6.0, 'text': 'To jest test. Sprawdzamy paragrafy.', 'words': [{'word':'To','start':3.5,'end':3.6},{'word':'jest','start':3.6,'end':3.8},{'word':'test.','start':3.8,'end':4.1},{'word':'Sprawdzamy','start':4.2,'end':4.7},{'word':'paragrafy.','start':4.7,'end':6.0}]}
    ]
    pars = paragraphize_segments(sample_segments, silence_threshold=1.0, max_chars=80, break_on_speaker=False)
    for i,p in enumerate(pars):
        print(f"PAR {i}: {p['start']:.2f}-{p['end']:.2f} | {p['text']}")

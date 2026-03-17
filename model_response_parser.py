import ast
import json
import re
from typing import Any, Optional


def strip_markdown_fences(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_json_loose(text: str) -> Optional[Any]:
    t = strip_markdown_fences(text)
    if not t:
        return None

    try:
        return json.loads(t)
    except Exception:
        pass

    try:
        return ast.literal_eval(t)
    except Exception:
        pass

    return None


def parse_list_response(text: str) -> Optional[list]:
    parsed = parse_json_loose(text)
    if isinstance(parsed, list):
        return parsed

    t = strip_markdown_fences(text)
    if not t:
        return None

    # Try to recover list from first bracketed fragment.
    candidates = re.findall(r"(\[[\s\S]*?\])", t, re.DOTALL)
    for cand in candidates:
        parsed = parse_json_loose(cand)
        if isinstance(parsed, list):
            return parsed

    # Fallback for JSON-ish one-item-per-line arrays.
    out = []
    for raw_line in t.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in ("[", "]"):
            continue
        if line.startswith("["):
            line = line[1:].lstrip()
        if line.endswith("]"):
            line = line[:-1].rstrip()
        if line.endswith(","):
            line = line[:-1].rstrip()

        if len(line) >= 2 and ((line[0] == '"' and line[-1] == '"') or (line[0] == "'" and line[-1] == "'")):
            line = line[1:-1]

        line = line.replace('\\"', '"').replace("\\'", "'").strip()
        if line:
            out.append(line)

    return out if out else None


def parse_dict_response(text: str) -> Optional[dict]:
    parsed = parse_json_loose(text)
    if isinstance(parsed, dict):
        return parsed

    t = strip_markdown_fences(text)
    if not t:
        return None

    m = re.search(r"(\{[\s\S]*\})", t)
    if not m:
        return None

    parsed = parse_json_loose(m.group(1))
    return parsed if isinstance(parsed, dict) else None

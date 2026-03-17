import os
from dataclasses import dataclass
from typing import Optional

from docx import Document

from config import downloads_dir
from file_utils import save_txt, save_docx, save_srt, load_srt


@dataclass
class TextArtifact:
    text: str
    segments: Optional[list] = None


def read_artifact(path: str, ext: str) -> TextArtifact:
    fmt = str(ext or "").lower().lstrip(".")

    if fmt == "txt":
        with open(path, "r", encoding="utf-8") as f:
            return TextArtifact(text=f.read(), segments=None)

    if fmt == "docx":
        doc = Document(path)
        txt = "\n\n".join([p.text for p in doc.paragraphs if p.text and p.text.strip()])
        return TextArtifact(text=txt, segments=None)

    if fmt == "srt":
        txt, segs = load_srt(path)
        return TextArtifact(text=txt, segments=segs)

    return TextArtifact(text="", segments=None)


def write_artifact(text: str, segments: Optional[list], path: str, ext: str) -> None:
    fmt = str(ext or "").lower().lstrip(".")

    if fmt == "txt":
        save_txt(text or "", path)
        return

    if fmt == "docx":
        save_docx(text or "", path)
        return

    if fmt == "srt":
        save_srt(segments or [], path)
        return


def write_to_output(base_name: str, suffix: str, ext: str, text: str = "", segments: Optional[list] = None) -> str:
    out_path = os.path.join(downloads_dir, f"{base_name}_{suffix}.{ext}")
    write_artifact(text=text, segments=segments, path=out_path, ext=ext)
    return out_path

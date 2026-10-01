from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import ast
import json
import re

from pypdf import PdfReader


@dataclass
class RawDocument:
    external_id: str
    title: str
    section: str | None
    text: str
    metadata: dict | None = None


def clean_caption_text(text: str) -> str:
    text = re.sub(r"^WEBVTT.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^NOTE.*(?:\n.*)*?(?=\n\n|\Z)", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\d{2}:\d{2}(?:\.\d{3})?\s*-->\s*\d{2}:\d{2}(?:\.\d{3})?.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def parse_structured_literal(raw: str) -> dict:
    """Accept strict JSON or a Python-dict-like pasted object without executing arbitrary code."""
    raw = raw.strip()
    if not raw:
        raise ValueError("Structured input is empty")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Input is neither valid JSON nor a safe Python dictionary literal") from exc
    if not isinstance(obj, dict):
        raise ValueError("Structured input must be an object/dictionary")
    return obj


def _split_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if blocks:
        return blocks
    return [re.sub(r"\s+", " ", text).strip()] if text.strip() else []


def _split_long(block: str, size: int) -> list[str]:
    if len(block) <= size:
        return [block]
    sentences = re.split(r"(?<=[.!?])\s+", block)
    if len(sentences) <= 1:
        words = block.split()
        out, current = [], []
        for word in words:
            candidate = (" ".join(current + [word])).strip()
            if current and len(candidate) > size:
                out.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            out.append(" ".join(current))
        return out
    out, current = [], ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > size:
            out.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def chunk_text(text: str, size: int = 1800, overlap: int = 220) -> list[str]:
    """Structure-aware chunking: paragraph/sentence boundaries first, then bounded overlap."""
    paragraphs = _split_paragraphs(text)
    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_split_long(paragraph, size))
    out: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        candidate = f"{current}\n{piece}"
        if len(candidate) <= size:
            current = candidate
        else:
            out.append(current.strip())
            tail = current[-overlap:].strip() if overlap else ""
            current = f"{tail}\n{piece}".strip() if tail else piece
            if len(current) > size:
                out.extend(_split_long(current, size)[:-1])
                current = _split_long(current, size)[-1]
    if current.strip():
        out.append(current.strip())
    return [x for x in out if x]


def parse_file(path: Path) -> list[RawDocument]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(RawDocument(f"{path.name}:p{i}", path.stem, f"page {i}", text))
        return docs
    if suffix in {".txt", ".md"}:
        return [RawDocument(path.name, path.stem, None, path.read_text(encoding="utf-8", errors="ignore"))]
    if suffix in {".srt", ".vtt"}:
        clean = clean_caption_text(path.read_text(encoding="utf-8", errors="ignore"))
        return [RawDocument(path.name, path.stem, "manual transcript", clean)]
    if suffix == ".json":
        payload = parse_structured_literal(path.read_text(encoding="utf-8", errors="ignore"))
        text = payload.get("transcript") or payload.get("text") or payload.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("JSON/dict source must contain transcript, text, or content")
        title = str(payload.get("name") or payload.get("title") or path.stem)
        return [RawDocument(path.name, title, payload.get("section"), text, payload)]
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [RawDocument(path.name, path.stem, None, text)]
    raise ValueError(f"Unsupported file type: {suffix}")


def youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname in {"youtu.be"}:
        return parsed.path.strip("/").split("/")[0] or None
    if parsed.hostname and "youtube.com" in parsed.hostname:
        q = parse_qs(parsed.query).get("v")
        if q:
            return q[0]
        m = re.search(r"/(?:shorts|embed)/([^/?]+)", parsed.path)
        return m.group(1) if m else None
    return None


def youtube_is_playlist(url: str) -> bool:
    return bool(parse_qs(urlparse(url).query).get("list"))


def list_youtube_videos(url: str) -> list[tuple[str, str]]:
    if not youtube_is_playlist(url):
        vid = youtube_video_id(url)
        if not vid:
            raise ValueError("Could not parse YouTube video URL")
        return [(vid, vid)]
    try:
        from yt_dlp import YoutubeDL
        with YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return [(entry["id"], entry.get("title") or entry["id"]) for entry in info.get("entries", []) if entry and entry.get("id")]
    except Exception as exc:
        raise RuntimeError(f"Playlist expansion failed: {exc}") from exc


def fetch_youtube_transcript(video_id: str, languages=("en", "tr")) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError("Install youtube-transcript-api to ingest YouTube transcripts") from exc
    api = YouTubeTranscriptApi()
    last_err = None
    for lang in languages:
        try:
            transcript = api.fetch(video_id, languages=[lang])
            return " ".join(sn.text for sn in transcript)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Transcript unavailable for {video_id}: {last_err}")

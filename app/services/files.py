import hashlib
from pathlib import Path
from fastapi import UploadFile
from ..config import settings


def safe_filename(filename: str | None) -> str:
    name = Path(filename or "upload.txt").name.strip()
    return name or "upload.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_upload(file: UploadFile) -> tuple[Path, str]:
    filename = safe_filename(file.filename)
    target = settings.upload_path / filename
    if target.exists():
        target = settings.upload_path / f"{hashlib.sha256(filename.encode()).hexdigest()[:8]}_{filename}"
    with target.open("wb") as out:
        while True:
            block = file.file.read(1024 * 1024)
            if not block:
                break
            out.write(block)
    return target, sha256_file(target)


def save_text_content(name: str, text: str) -> tuple[Path, str]:
    filename = safe_filename(name)
    if Path(filename).suffix.lower() not in {".txt", ".md", ".srt", ".vtt"}:
        filename = f"{filename}.txt"
    target = settings.upload_path / filename
    if target.exists():
        target = settings.upload_path / f"{hashlib.sha256(filename.encode()).hexdigest()[:8]}_{filename}"
    target.write_text(text, encoding="utf-8")
    return target, sha256_file(target)

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path, PurePosixPath


class UnsafeZipError(ValueError):
    """Raised when an archive contains an unsafe extraction target."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_zip_members(zip_path: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts:
                raise UnsafeZipError(f'Unsafe ZIP entry: {info.filename}')
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise UnsafeZipError(f'ZIP symlink entry is not allowed: {info.filename}')
            names.append(info.filename)
    return names


def sha256_zip_member(zf: zipfile.ZipFile, member: str) -> str:
    h = hashlib.sha256()
    with zf.open(member, 'r') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_extract_zip(zip_path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    validate_zip_members(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            destination = target / PurePosixPath(info.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            with zf.open(info, 'r') as source, destination.open('wb') as output:
                shutil.copyfileobj(source, output)

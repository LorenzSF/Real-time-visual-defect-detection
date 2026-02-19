from __future__ import annotations

import zipfile
from pathlib import Path


def _validate_zip_member(member_name: str, extract_dir: Path) -> None:
    member_path = Path(member_name)
    if member_path.is_absolute():
        raise ValueError(f"Unsafe ZIP member (absolute path): {member_name}")

    resolved_target = (extract_dir / member_path).resolve()
    try:
        resolved_target.relative_to(extract_dir)
    except ValueError as exc:
        raise ValueError(f"Unsafe ZIP member (path traversal): {member_name}") from exc


def extract_zip(zip_path: str | Path, extract_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    extract_dir = Path(extract_dir).resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            _validate_zip_member(info.filename, extract_dir)
        zf.extractall(extract_dir)

    return extract_dir
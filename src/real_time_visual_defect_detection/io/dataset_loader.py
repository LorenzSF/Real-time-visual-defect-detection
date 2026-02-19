from __future__ import annotations

from pathlib import Path
from typing import List

from .zip_ingest import extract_zip


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root_dir: str | Path) -> List[Path]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    paths = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            paths.append(p)

    return sorted(paths)


def resolve_dataset(source_type: str, path: str, extract_dir: str) -> List[Path]:
    """
    Returns a list of image paths.
    - source_type="zip": extracts zip to extract_dir then lists images
    - source_type="folder": lists images under path
    """
    if source_type == "zip":
        extracted = extract_zip(path, extract_dir)
        return list_images(extracted)

    if source_type == "folder":
        return list_images(path)

    raise ValueError(f"Unknown source_type: {source_type}. Use 'zip' or 'folder'.")

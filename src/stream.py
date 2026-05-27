import json
import random
import time
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import numpy as np
from PIL import Image

from .models import Model
from .schemas import (
    DatasetEntry,
    Frame,
    OfflineSplit,
    OfflineSplitConfig,
    StreamConfig,
    WarmupConfig,
)


Entry = Tuple[Path, int, str]


def build_warmup_stream(cfg: StreamConfig, warmup_steps: int) -> Iterator[Frame]:
    """Yield the first `warmup_steps` images from the sorted input listing.

    The warmup slice ignores `cfg.shuffle` so the OK-first ordering produced
    by `data/prepare_dataset.py` is preserved.
    """
    entries = _discover_input_images(cfg)
    selected = entries[:warmup_steps]
    yield from _yield_frames(selected)


def build_stream(
    cfg: StreamConfig, warmup_steps: int, presorted_prefix: int = 0
) -> Iterator[Frame]:
    """Yield the post-warmup inference stream from the configured input folder.

    `entries[warmup_steps:]` is the inference slice. When `cfg.shuffle` is
    true, only `inference_entries[presorted_prefix:]` is shuffled in place;
    the first `presorted_prefix` inference frames keep their sorted order.
    The caller (main.py) passes `cfg.metrics.calibration_steps` here so the
    threshold-calibration window draws from the sorted, OK-first prefix
    before the rest of the stream is randomized.
    """
    entries = _discover_input_images(cfg)
    inference_entries = entries[warmup_steps:]
    if cfg.shuffle and presorted_prefix < len(inference_entries):
        tail = inference_entries[presorted_prefix:]
        random.shuffle(tail)
        inference_entries = inference_entries[:presorted_prefix] + tail
    if cfg.max_frames is not None:
        inference_entries = inference_entries[: cfg.max_frames]
    if not inference_entries:
        raise FileNotFoundError(
            f"no inference frames left under {Path(cfg.input_path).resolve()} "
            f"after reserving {warmup_steps} frames for warmup"
        )
    yield from _yield_frames(inference_entries)


def warmup(model: Model, stream: Iterator[Frame], cfg: WarmupConfig) -> List[Frame]:
    """Consume `cfg.warmup_steps` frames and call `model.fit_warmup`."""
    if not hasattr(model, "fit_warmup"):
        raise TypeError(f"model {type(model).__name__} has no fit_warmup() method")

    frames: List[Frame] = []
    for i, frame in enumerate(stream):
        if i >= cfg.warmup_steps:
            break
        frames.append(frame)
    if not frames:
        raise RuntimeError("no frames available for warmup")
    if len(frames) < cfg.warmup_steps:
        raise RuntimeError(
            f"warmup requires {cfg.warmup_steps} frames, got only {len(frames)}"
        )
    model.fit_warmup(frames)
    return frames


def discover_dataset_entries(cfg: StreamConfig) -> List[DatasetEntry]:
    return [
        DatasetEntry(path=str(path), label=label, image_id=image_id)
        for path, label, image_id in _discover_input_images(cfg)
    ]


def build_offline_split(
    cfg: StreamConfig, split_cfg: OfflineSplitConfig, seed: int
) -> OfflineSplit:
    entries = discover_dataset_entries(cfg)
    if cfg.max_frames is not None:
        entries = entries[: cfg.max_frames]

    unknown = [entry.image_id for entry in entries if entry.label not in (0, 1)]
    if unknown:
        sample = ", ".join(unknown[:5])
        raise ValueError(
            "offline mode requires every image to have an OK/NG label in labels.json; "
            f"first missing labels: {sample}"
        )

    if split_cfg.stratify:
        ok_entries = [entry for entry in entries if entry.label == 0]
        ng_entries = [entry for entry in entries if entry.label == 1]
        ok_train, ok_val, ok_test = _split_group(ok_entries, split_cfg, seed)
        ng_train, ng_val, ng_test = _split_group(ng_entries, split_cfg, seed + 1)
        if split_cfg.train_on_good_only:
            train = ok_train
            test = ng_train + ok_test + ng_test
        else:
            train = ok_train + ng_train
            test = ok_test + ng_test
        val = ok_val + ng_val
    else:
        train, val, test = _split_group(entries, split_cfg, seed)
        if split_cfg.train_on_good_only:
            rejected = [entry for entry in train if entry.label == 1]
            train = [entry for entry in train if entry.label == 0]
            test = rejected + test

    train_goods = sum(1 for entry in train if entry.label == 0)
    if train_goods < split_cfg.min_train_goods:
        raise RuntimeError(
            "offline split has fewer OK train samples than requested: "
            f"{train_goods} < {split_cfg.min_train_goods}"
        )
    if not train or not val or not test:
        raise RuntimeError(
            "offline split produced an empty train, val, or test split "
            f"(train={len(train)}, val={len(val)}, test={len(test)})"
        )
    if not _has_both_labels(val):
        raise RuntimeError("offline validation split must contain both OK and NG labels")
    if not _has_both_labels(test):
        raise RuntimeError("offline test split must contain both OK and NG labels")

    train_split = list(train)
    val_split = list(val)
    test_split = list(test)
    random.Random(seed + 100).shuffle(train_split)
    random.Random(seed + 101).shuffle(val_split)
    random.Random(seed + 102).shuffle(test_split)
    return OfflineSplit(train=train_split, val=val_split, test=test_split)


def frames_from_entries(entries: List[DatasetEntry]) -> Iterator[Frame]:
    for index, entry in enumerate(entries):
        yield Frame(
            image=_load_image(Path(entry.path)),
            label=entry.label,
            timestamp=time.time(),
            source_id=entry.path,
            image_id=entry.image_id,
            index=index,
        )


def _split_group(
    entries: List[DatasetEntry], split_cfg: OfflineSplitConfig, seed: int
) -> tuple[List[DatasetEntry], List[DatasetEntry], List[DatasetEntry]]:
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_val = int(round(n_total * split_cfg.val_ratio))
    n_test = int(round(n_total * split_cfg.test_ratio))
    n_val = min(n_val, n_total)
    n_test = min(n_test, max(0, n_total - n_val))
    val = shuffled[:n_val]
    test = shuffled[n_val : n_val + n_test]
    train = shuffled[n_val + n_test :]
    return train, val, test


def _has_both_labels(entries: List[DatasetEntry]) -> bool:
    labels = {entry.label for entry in entries}
    return 0 in labels and 1 in labels


def _yield_frames(entries: List[Entry]) -> Iterator[Frame]:
    for index, (img_path, label, image_id) in enumerate(entries):
        yield Frame(
            image=_load_image(img_path),
            label=label,
            timestamp=time.time(),
            source_id=str(img_path.as_posix()),
            image_id=image_id,
            index=index,
        )


def _discover_input_images(cfg: StreamConfig) -> List[Entry]:
    root = Path(cfg.input_path)
    if not root.is_dir():
        raise FileNotFoundError(f"stream.input_path is not a directory: {root}")

    subdirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    if subdirs:
        raise ValueError(
            "stream.input_path must be a flat folder of images. "
            "Subfolders are not supported; see README.md for the required input format. "
            f"Found subfolders: {subdirs}"
        )

    extensions = {ext.lower() for ext in cfg.extensions}
    labels = _load_labels(root)
    paths = list(_iter_images(root, extensions))
    if not paths:
        raise FileNotFoundError(
            f"no images with extensions {sorted(extensions)} found under {root.resolve()}"
        )

    seen_ids: Set[str] = set()
    entries: List[Entry] = []
    for img_path in paths:
        image_id = img_path.stem
        if image_id in seen_ids:
            raise ValueError(
                f"duplicate image id {image_id!r} under {root}; "
                "filenames without extension must be unique"
            )
        seen_ids.add(image_id)
        entries.append((img_path, labels.get(image_id, -1), image_id))

    unknown_label_ids = sorted(set(labels) - seen_ids)
    if unknown_label_ids:
        raise ValueError(
            "labels.json contains ids that do not match any input image: "
            f"{unknown_label_ids}"
        )

    return entries


def _load_labels(root: Path) -> dict[str, int]:
    path = root / "labels.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError("labels.json must be an object mapping image_id to 'OK' or 'NG'")

    labels: dict[str, int] = {}
    for image_id, label in raw.items():
        if not isinstance(image_id, str) or not image_id:
            raise TypeError("labels.json keys must be non-empty image_id strings")
        if label == "OK":
            labels[image_id] = 0
        elif label == "NG":
            labels[image_id] = 1
        else:
            raise ValueError(
                "labels.json values must be exactly 'OK' or 'NG', "
                f"got {label!r} for image_id {image_id!r}"
            )
    return labels


def _iter_images(root: Path, extensions: Set[str]) -> Iterator[Path]:
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        yield path


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))

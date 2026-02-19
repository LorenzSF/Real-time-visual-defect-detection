from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from real_time_visual_defect_detection.core import load_config
from real_time_visual_defect_detection.pipelines.run_pipeline import run_pipeline


DEFAULT_HISTORY_FILE = Path("data") / ".dataset_path_history.json"
MAX_HISTORY_ENTRIES = 10


def _clean_path_text(raw: str) -> str:
    return raw.strip().strip('"').strip("'")


def _canonical_path(path_text: str) -> str:
    path = Path(path_text).expanduser()
    try:
        path = path.resolve(strict=False)
    except OSError:
        pass
    return str(path)


def _load_history(history_file: Path) -> List[str]:
    if not history_file.exists():
        return []

    try:
        data = json.loads(history_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, str) and item.strip()]


def _save_history(history_file: Path, paths: List[str]) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.write_text(json.dumps(paths, indent=2), encoding="utf-8")


def _update_history(history_file: Path, selected_path: str) -> None:
    selected = _canonical_path(selected_path)
    selected_key = selected.lower()

    updated: List[str] = [selected]
    for item in _load_history(history_file):
        item_norm = _canonical_path(item)
        if item_norm.lower() == selected_key:
            continue
        updated.append(item_norm)
        if len(updated) >= MAX_HISTORY_ENTRIES:
            break

    _save_history(history_file, updated)


def _prompt_dataset_path(current_path: str | None, history_file: Path) -> str:
    options: List[str] = []
    seen = set()

    if current_path:
        first = _canonical_path(current_path)
        options.append(first)
        seen.add(first.lower())

    for p in _load_history(history_file):
        p_norm = _canonical_path(p)
        p_key = p_norm.lower()
        if p_key in seen:
            continue
        options.append(p_norm)
        seen.add(p_key)

    print("\nDataset path selection")
    for idx, path_text in enumerate(options, start=1):
        source = "config" if idx == 1 and current_path else "recent"
        status = "exists" if Path(path_text).exists() else "missing"
        print(f"  {idx}. {path_text} [{source}, {status}]")
    print("  N. Enter a new path")

    while True:
        try:
            choice = input("Choose dataset path (number or N): ").strip()
        except EOFError as exc:
            raise ValueError("Interactive selection aborted (no input available).") from exc

        if choice.lower() in {"n", "new"} or (not options and choice == ""):
            entered = _clean_path_text(input("Enter dataset ZIP/folder path: "))
            if not entered:
                print("Path cannot be empty.")
                continue
            selected = _canonical_path(entered)
            if not Path(selected).exists():
                print(f"Path not found: {selected}")
                continue
            _update_history(history_file, selected)
            return selected

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                selected = options[idx - 1]
                if not Path(selected).exists():
                    print(f"Selected path does not exist: {selected}")
                    continue
                _update_history(history_file, selected)
                return selected

        print("Invalid choice. Type a number from the list or N.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional override for cfg['dataset']['path'] (useful for local ZIP paths).",
    )
    p.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="Optional override for cfg['dataset']['extract_dir'].",
    )
    p.add_argument(
        "--choose-dataset",
        action="store_true",
        help="Interactively choose dataset path from config/history, or enter a new one.",
    )
    p.add_argument(
        "--history-file",
        type=str,
        default=str(DEFAULT_HISTORY_FILE),
        help="Path to JSON file that stores recently used dataset paths.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    cfg.setdefault("dataset", {})

    if args.dataset_path is not None:
        cfg["dataset"]["path"] = args.dataset_path
    if args.extract_dir is not None:
        cfg["dataset"]["extract_dir"] = args.extract_dir

    history_file = Path(args.history_file)
    dataset_path = cfg["dataset"].get("path")
    should_prompt = bool(args.choose_dataset)

    if dataset_path is None:
        should_prompt = True
    else:
        dataset_exists = Path(str(dataset_path)).expanduser().exists()
        if not dataset_exists and sys.stdin.isatty():
            print(f"Configured dataset path does not exist: {dataset_path}")
            should_prompt = True
        if not dataset_exists and not sys.stdin.isatty() and not should_prompt:
            raise FileNotFoundError(
                f"Dataset path not found: {dataset_path}. "
                "Use --dataset-path or --choose-dataset to select a valid path."
            )

    if should_prompt:
        cfg["dataset"]["path"] = _prompt_dataset_path(
            str(cfg["dataset"].get("path")) if cfg["dataset"].get("path") is not None else None,
            history_file=history_file,
        )

    out_dir = run_pipeline(cfg)
    print(f"Run complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

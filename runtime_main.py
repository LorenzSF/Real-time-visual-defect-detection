from __future__ import annotations

import argparse
from pathlib import Path

from streaming_input.app import StreamingInputApp
from streaming_input.settings import DEFAULT_SETTINGS_FILE, load_settings
from corruptions.corruption_registry import (
    SEVERITY_LEVELS,
    available_corruptions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_SETTINGS_FILE),
        help="Path to the runtime YAML settings file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override cfg['artifact']['model_name']. Manual selection only — the "
            "best benchmark model is never picked automatically."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Override cfg['artifact']['run_dir'] with a specific benchmark run "
            "directory. Pass 'latest' (or omit) to use the most recent run."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional override for run.max_frames.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override cfg['input']['root_dir'] with the live folder to stream.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the input folder indefinitely instead of stopping after one pass.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Optional override for run.target_fps.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable the live dashboard server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional override for web.port.",
    )
    parser.add_argument(
        "--fit-policy",
        type=str,
        default=None,
        choices=("auto", "historical_fit", "skip_fit"),
        help="Override cfg['artifact']['fit_policy'].",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Override the historical dataset path embedded in benchmark_summary.json "
            "when the runtime reconstructs the train/val/test split."
        ),
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help=(
            "Override the historical extract_dir embedded in benchmark_summary.json. "
            "Defaults to --dataset-path when omitted for folder datasets."
        ),
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default=None,
        choices=list(available_corruptions()),
        help="Apply this corruption to streamed frames. Enables cfg['corruption'].",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=None,
        choices=list(SEVERITY_LEVELS),
        help="Severity (1..5) used together with --corruption.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_settings(Path(args.config))
    if args.max_frames is not None:
        cfg.setdefault("run", {})
        cfg["run"]["max_frames"] = args.max_frames
    if args.target_fps is not None:
        cfg.setdefault("run", {})
        cfg["run"]["target_fps"] = float(args.target_fps)
    if args.input_dir is not None or args.loop:
        cfg.setdefault("input", {})
        if args.input_dir is not None:
            cfg["input"]["root_dir"] = args.input_dir
        if args.loop:
            cfg["input"]["loop"] = True
    if args.no_web:
        cfg.setdefault("web", {})
        cfg["web"]["enabled"] = False
    if args.port is not None:
        cfg.setdefault("web", {})
        cfg["web"]["port"] = int(args.port)

    # Manual model selection via CLI — never auto-pick the best benchmark model.
    if (
        args.model is not None
        or args.run_dir is not None
        or args.fit_policy is not None
        or args.dataset_path is not None
        or args.extract_dir is not None
    ):
        artifact = dict(cfg.get("artifact", {}))
        if args.model is not None:
            artifact["model_name"] = args.model
        if args.run_dir is not None:
            artifact["run_dir"] = args.run_dir
        if args.fit_policy is not None:
            artifact["fit_policy"] = args.fit_policy
        if args.dataset_path is not None:
            artifact["dataset_path_override"] = args.dataset_path
        if args.extract_dir is not None:
            artifact["extract_dir_override"] = args.extract_dir
        cfg["artifact"] = artifact

    # Stash the corruption choice in cfg; the streaming app picks it up in
    # block 2.2. Keeping the CLI surface symmetric with main.py now avoids a
    # later breaking change to scripts that already use these flags.
    if args.corruption is not None or args.severity is not None:
        corr = dict(cfg.get("corruption", {}))
        if args.corruption is not None:
            corr["type"] = args.corruption
            corr["enabled"] = True
        if args.severity is not None:
            corr["severity"] = args.severity
        corr.setdefault("enabled", True)
        cfg["corruption"] = corr

    app = StreamingInputApp(cfg)
    session_dir = app.run()
    print(f"Runtime session saved to: {session_dir}")


if __name__ == "__main__":
    main()

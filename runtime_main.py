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
        "--max-frames",
        type=int,
        default=None,
        help="Optional override for run.max_frames.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable the live dashboard server.",
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
    if args.no_web:
        cfg.setdefault("web", {})
        cfg["web"]["enabled"] = False

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

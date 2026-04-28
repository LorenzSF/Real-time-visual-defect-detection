from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark_AD.models import available_models, model_dependencies
from benchmark_AD.pipeline import load_config, run_pipeline
from corruptions.corruption_registry import (
    SEVERITY_LEVELS,
    available_corruptions,
)


DEFAULT_CONFIG_FILE = Path("src") / "benchmark_AD" / "default.yaml"


def _install_hint(module_name: str) -> str:
    hints = {
        "lightning": "lightning",
        "lightning.pytorch": "lightning",
        "anomalib": "anomalib",
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "FrEIA": "FrEIA",
        "kornia": "kornia",
        "transformers": "transformers",
    }
    return hints.get(module_name, module_name.split(".", 1)[0])


def _module_issue(module_name: str) -> Optional[str]:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or module_name
        return f"missing dependency '{missing}' (pip install {_install_hint(missing)})"
    except Exception as exc:
        return f"import error: {exc}"
    return None


def _model_preflight_checks(model_name: str) -> List[Tuple[str, Optional[str]]]:
    return [(mod, _module_issue(mod)) for mod in model_dependencies(model_name)]


def _model_runtime_issue(model_name: str) -> Optional[str]:
    for module_name, issue in _model_preflight_checks(model_name):
        if issue is not None:
            return f"{module_name}: {issue}"
    return None


def _infer_source_type(path_text: str) -> str:
    path = Path(path_text)
    return "zip" if path.is_file() and path.suffix.lower() == ".zip" else "folder"


def _base_model_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = dict(cfg.get("model", {})) if isinstance(cfg.get("model"), dict) else {}
    base.pop("name", None)
    return base


def _apply_single_model(cfg: Dict[str, Any], model_name: str) -> None:
    """Pin the pipeline to a single model, preserving its per-kind sub-config.

    When ``benchmark.models[]`` contains an entry matching ``model_name``,
    that entry's full config (e.g. PaDiM-specific ``anomalib`` overrides) is
    used as the base — otherwise the global ``cfg['model']`` block is used.
    Without this, --model would silently drop per-model YAML overrides and
    e.g. PaDiM would inherit PatchCore's wide_resnet50_2 backbone and OOM
    on the covariance step.
    """
    base: Optional[Dict[str, Any]] = None
    bench_models = cfg.get("benchmark", {}).get("models")
    if isinstance(bench_models, list):
        for m in bench_models:
            if isinstance(m, dict) and m.get("name") == model_name:
                base = {k: v for k, v in m.items() if k != "name"}
                break
    if base is None:
        base = _base_model_cfg(cfg)

    entry = {**base, "name": model_name}
    cfg["model"] = dict(entry)
    bench = dict(cfg.get("benchmark", {})) if isinstance(cfg.get("benchmark"), dict) else {}
    bench["models"] = [dict(entry)]
    cfg["benchmark"] = bench


def _apply_all_models(cfg: Dict[str, Any]) -> List[str]:
    """Expand benchmark.models with every registry entry that passes preflight.

    Returns the list of skipped models along with the reason, so the caller can
    surface the information in the run log without blocking the job.
    """
    base = _base_model_cfg(cfg)
    entries: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for name in available_models():
        issue = _model_runtime_issue(name)
        if issue is not None:
            skipped.append(f"{name} ({issue})")
            continue
        entries.append({**base, "name": name})

    if not entries:
        raise RuntimeError(
            "No models are selectable in the current environment. "
            f"Skipped: {'; '.join(skipped) if skipped else 'none'}."
        )

    bench = dict(cfg.get("benchmark", {})) if isinstance(cfg.get("benchmark"), dict) else {}
    bench["models"] = entries
    cfg["benchmark"] = bench
    return skipped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the anomaly-detection benchmark pipeline."
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_FILE),
        help="Path to YAML/JSON config. May use `_extends` to overlay a base config.",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override for cfg['dataset']['path'] (folder or .zip).",
    )
    p.add_argument(
        "--extract-dir",
        type=str,
        default=None,
        help="Override for cfg['dataset']['extract_dir'] (ZIP datasets only).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pin the run to a single registered model (e.g. anomalib_patchcore).",
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark every registered model that passes dependency preflight.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override cfg['run']['run_name']; shapes the output subdir name.",
    )
    p.add_argument(
        "--corruption",
        type=str,
        default=None,
        choices=list(available_corruptions()),
        help="Apply this corruption to test images. Enables cfg['corruption'].",
    )
    p.add_argument(
        "--severity",
        type=int,
        default=None,
        choices=list(SEVERITY_LEVELS),
        help="Severity (1..5) used when --corruption is set or in the config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.model and args.all_models:
        raise SystemExit("Use --model or --all-models, not both.")

    cfg = load_config(Path(args.config))
    cfg.setdefault("dataset", {})
    cfg.setdefault("run", {})

    if args.dataset_path is not None:
        cfg["dataset"]["path"] = args.dataset_path
        cfg["dataset"]["source_type"] = _infer_source_type(args.dataset_path)
    if args.extract_dir is not None:
        cfg["dataset"]["extract_dir"] = args.extract_dir
    if args.run_name is not None:
        cfg["run"]["run_name"] = args.run_name

    # CLI corruption flags override the YAML; passing --corruption alone enables
    # the section, passing --severity alone tweaks the configured corruption.
    if args.corruption is not None or args.severity is not None:
        corr = dict(cfg.get("corruption", {}))
        if args.corruption is not None:
            corr["type"] = args.corruption
            corr["enabled"] = True
        if args.severity is not None:
            corr["severity"] = args.severity
        corr.setdefault("enabled", True)
        cfg["corruption"] = corr

    dataset_path = cfg["dataset"].get("path")
    if dataset_path is None:
        raise SystemExit(
            "dataset.path is not set. Provide it in the config or via --dataset-path."
        )
    if not Path(str(dataset_path)).expanduser().exists():
        raise SystemExit(f"Dataset path not found: {dataset_path}")

    if args.all_models:
        skipped = _apply_all_models(cfg)
        if skipped:
            print("[main] Skipped unavailable models: " + "; ".join(skipped), file=sys.stderr)
    elif args.model is not None:
        if args.model not in available_models():
            supported = ", ".join(available_models())
            raise SystemExit(f"Unknown model '{args.model}'. Supported: {supported}.")
        issue = _model_runtime_issue(args.model)
        if issue is not None:
            raise SystemExit(f"Model '{args.model}' preflight failed: {issue}.")
        _apply_single_model(cfg, args.model)

    out_dir = run_pipeline(cfg)
    print(f"Run complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

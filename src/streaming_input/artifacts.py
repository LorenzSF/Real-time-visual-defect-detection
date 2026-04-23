from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .contracts import RuntimeArtifact


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _resolve_model_cfg(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    bench_cfg = cfg.get("benchmark", {})
    models = bench_cfg.get("models", [])
    if isinstance(models, list):
        for entry in models:
            if isinstance(entry, dict) and entry.get("name") == model_name:
                return dict(entry)

    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        resolved = dict(model_cfg)
        resolved["name"] = model_name
        return resolved
    raise ValueError(f"Unable to resolve model configuration for: {model_name}")


def _summary_rows(summary_doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary_doc.get("models", [])
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("Invalid benchmark summary: missing model rows.")
    return [row for row in rows if isinstance(row, dict)]


def _select_summary_row(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("model")) == model_name:
            return row
    raise ValueError(f"Model '{model_name}' not found in benchmark summary.")


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime_artifact(
    run_dir: str | Path,
    model_name: Optional[str] = None,
) -> RuntimeArtifact:
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {summary_path}")

    summary_doc = _load_json(summary_path, default={})
    if not isinstance(summary_doc, dict):
        raise ValueError(f"Invalid benchmark summary: {summary_path}")
    summary_rows = _summary_rows(summary_doc)

    if model_name is None:
        model_name = str(summary_rows[0].get("model", ""))
    if not model_name:
        raise ValueError("Unable to resolve runtime model name from benchmark summary.")

    summary_row = _select_summary_row(summary_rows, model_name=model_name)
    validation_predictions_path = run_dir / f"validation_predictions_{_safe_name(model_name)}.json"
    model_cfg = summary_row.get("model_cfg")
    if not isinstance(model_cfg, dict):
        model_cfg = _resolve_model_cfg(summary_doc, model_name=model_name)
    threshold = float(summary_row.get("threshold_used", model_cfg.get("threshold", 0.5)))

    return RuntimeArtifact(
        run_dir=run_dir,
        model_name=model_name,
        model_cfg=dict(model_cfg),
        runtime_cfg=dict(summary_doc.get("runtime", {})),
        dataset_cfg=dict(summary_doc.get("dataset", {})),
        preprocessing_cfg=dict(summary_doc.get("preprocessing", {})),
        threshold=threshold,
        validation_predictions_path=validation_predictions_path if validation_predictions_path.exists() else None,
    )

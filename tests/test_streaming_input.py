from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmark_AD.data import LabeledSample
import streaming_input.app as app_module
from streaming_input.settings import resolve_runtime_settings


def _frame(
    image: np.ndarray,
    path: Path,
    frame_id: int,
    label: int,
    defect_type: str | None,
) -> app_module._Frame:
    return app_module._Frame(
        frame_id=frame_id,
        timestamp_utc="2026-01-01T00:00:00+00:00",
        path=path,
        raw_image_bgr=image,
        model_input=image.astype(np.float32) / 255.0,
        label=label,
        defect_type=defect_type,
    )


def test_resolve_runtime_settings_warns_and_drops_legacy_sections():
    cfg = {
        "artifact": {"run_dir": "data/outputs/example"},
        "object_change": {"distance_threshold": 0.5},
        "calibration": {"min_frames": 20},
    }

    with pytest.warns(RuntimeWarning) as caught:
        resolved = resolve_runtime_settings(cfg)

    messages = [str(item.message) for item in caught]
    assert messages == [
        "Ignoring deprecated runtime config section 'object_change'.",
        "Ignoring deprecated runtime config section 'calibration'.",
    ]
    assert "object_change" not in resolved
    assert "calibration" not in resolved
    assert resolved["run"]["output_dir"] == "data/streaming_output"
    assert resolved["run"]["session_name"] == "streaming_output"


def test_interleave_by_label_mixes_good_bad_and_unlabeled():
    samples = [
        LabeledSample(path=Path("good_1.png"), label=0),
        LabeledSample(path=Path("good_2.png"), label=0),
        LabeledSample(path=Path("bad_1.png"), label=1),
        LabeledSample(path=Path("bad_2.png"), label=1),
        LabeledSample(path=Path("unknown_1.png"), label=-1),
    ]

    ordered = app_module._interleave_by_label(samples)
    ordered_paths = [sample.path.name for sample in ordered]
    assert ordered_paths == [
        "good_1.png",
        "bad_1.png",
        "unknown_1.png",
        "good_2.png",
        "bad_2.png",
    ]


def test_runtime_app_emits_model_suffixed_artifacts(tmp_path: Path, monkeypatch):
    input_root = tmp_path / "source_frames"
    input_root.mkdir()

    frames = [
        _frame(
            image=np.zeros((12, 12, 3), dtype=np.uint8),
            path=input_root / "good.png",
            frame_id=1,
            label=0,
            defect_type=None,
        ),
        _frame(
            image=np.full((12, 12, 3), 255, dtype=np.uint8),
            path=input_root / "bad.png",
            frame_id=2,
            label=1,
            defect_type="scratch",
        ),
    ]

    class _FakeFrameInference:
        init_calls: list[dict[str, object]] = []

        def __init__(
            self,
            run_dir: Path,
            model_name: str,
            fit_policy: str = "auto",
            dataset_path_override: str | None = None,
            extract_dir_override: str | None = None,
        ) -> None:
            self.run_dir = Path(run_dir)
            self.model_name = model_name
            self.fit_policy = fit_policy
            self.dataset_path_override = dataset_path_override
            self.extract_dir_override = extract_dir_override
            self.threshold = 0.5
            self.summary = {
                "run_name": "baseline",
                "models": [{"model": model_name, "threshold_used": self.threshold}],
            }
            self._scores = iter((0.2, 0.9))
            type(self).init_calls.append(
                {
                    "run_dir": self.run_dir,
                    "model_name": self.model_name,
                    "fit_policy": self.fit_policy,
                    "dataset_path_override": self.dataset_path_override,
                    "extract_dir_override": self.extract_dir_override,
                }
            )

        def predict(self, image: np.ndarray):
            score = next(self._scores)
            heatmap = np.ones((6, 6), dtype=np.float32) if score >= self.threshold else None
            return type(
                "Prediction",
                (),
                {
                    "anomaly_score": score,
                    "anomaly_map": heatmap,
                    "embedding": None,
                },
            )()

    def _fake_iter_folder_frames(**_: object):
        yield from frames

    monkeypatch.setattr(app_module, "FrameInference", _FakeFrameInference)
    monkeypatch.setattr(app_module, "_iter_folder_frames", _fake_iter_folder_frames)

    cfg = {
        "run": {
            "output_dir": str(tmp_path),
            "target_fps": 0.0,
            "max_frames": 2,
        },
        "artifact": {
            "run_dir": str(tmp_path),
            "model_name": "stub_model",
            "fit_policy": "historical_fit",
            "dataset_path_override": str(tmp_path / "history"),
            "extract_dir_override": str(tmp_path / "history_extract"),
        },
        "input": {
            "root_dir": str(input_root),
            "loop": False,
            "sequence_mode": "interleaved_labels",
        },
        "preprocessing": {
            "resize": {"enabled": False},
            "normalize": {"enabled": True},
        },
        "corruption": {
            "enabled": True,
            "type": "gaussian_blur",
            "severity": 3,
        },
        "web": {"enabled": False},
    }

    session_dir = app_module.StreamingInputApp(cfg).run()

    assert session_dir.parent == tmp_path
    assert session_dir.name.startswith("streaming_output_")

    benchmark_summary = json.loads((session_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    predictions = json.loads(
        (session_dir / "predictions_stub_model.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (session_dir / "live_status_stub_model.json").read_text(encoding="utf-8")
    )

    assert benchmark_summary["models"][0]["model"] == "stub_model"
    assert _FakeFrameInference.init_calls == [
        {
            "run_dir": tmp_path,
            "model_name": "stub_model",
            "fit_policy": "historical_fit",
            "dataset_path_override": str(tmp_path / "history"),
            "extract_dir_override": str(tmp_path / "history_extract"),
        }
    ]
    assert len(predictions) == 2
    assert [row["path"] for row in predictions] == [
        str(input_root / "good.png"),
        str(input_root / "bad.png"),
    ]
    assert predictions[0]["corruption_type"] == "gaussian_blur"
    assert predictions[0]["severity"] == 3
    assert predictions[0]["dataset"] == input_root.name
    assert status["decisions_emitted"] == 2
    assert status["frames_seen"] == 2
    assert status["fail_count"] == 1
    assert status["active_model"] == "stub_model"
    assert status["threshold"] == 0.5
    assert status["corruption_type"] == "gaussian_blur"
    assert status["severity"] == 3
    assert status["dataset"] == input_root.name
    assert status["latest_score"] == 0.9
    assert status["score_history"] == [0.2, 0.9]
    assert status["anomaly_rate"] == 0.5
    assert "rolling_fps_10" in status
    assert status["current_frame_path"] == "dashboard/current_frame.png"
    assert status["current_frame_version"] == 2
    assert status["embedding_enabled"] is False
    assert len(status["recent_fails"]) == 1
    assert (session_dir / "heatmaps" / "frame_000002.png").exists()
    assert (session_dir / "dashboard" / "current_frame.png").exists()
    assert not (session_dir / "predictions.json").exists()
    assert not (session_dir / "live_status.json").exists()
    for removed_key in (
        "state",
        "object_change_count",
        "no_decision_count",
        "classifier_confidence",
        "classifier_distance",
        "calibration_frames",
        "calibration_score_mean",
        "baseline_score_mean",
        "baseline_score_std",
    ):
        assert removed_key not in status

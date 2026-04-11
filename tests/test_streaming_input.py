from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark_AD.data import LabeledSample
import streaming_input.app as app_module
from streaming_input.contracts import FramePacket, ModelPrediction
from streaming_input.decision_engine import DecisionEngine
from streaming_input.input_handler import _interleave_by_label
from streaming_input.report_generator import RuntimeOutputWriter
from streaming_input.settings import resolve_runtime_settings


def _frame(image: np.ndarray, frame_id: int = 0) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_utc="2026-01-01T00:00:00+00:00",
        path=Path("sample.png"),
        raw_image_bgr=image,
        model_input=image.astype(np.float32) / 255.0,
        label=0,
        defect_type=None,
    )


def test_resolve_runtime_settings_warns_and_drops_legacy_sections():
    cfg = {
        "artifact": {"run_dir": "data/runs/example"},
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


def test_decision_engine_keeps_baseline_prediction_shape():
    engine = DecisionEngine()
    frame = _frame(np.zeros((16, 16, 3), dtype=np.uint8))
    prediction = ModelPrediction(
        model_name="stub_model",
        score=0.75,
        pred_is_anomaly=1,
        heatmap=np.ones((16, 16), dtype=np.float32),
    )

    record = engine.decide(frame=frame, prediction=prediction)
    row = record.to_dict()
    assert set(row) == {
        "model",
        "path",
        "label",
        "defect_type",
        "score",
        "pred_is_anomaly",
        "heatmap_path",
    }
    assert row["pred_is_anomaly"] == 1


def test_output_writer_saves_predictions_and_heatmap(tmp_path: Path):
    writer = RuntimeOutputWriter(session_dir=tmp_path)
    image = np.full((24, 24, 3), 100, dtype=np.uint8)
    frame = _frame(image=image, frame_id=7)
    prediction = ModelPrediction(
        model_name="stub_model",
        score=0.9,
        pred_is_anomaly=1,
        heatmap=np.ones((24, 24), dtype=np.float32),
    )

    heatmap_path = writer.save_heatmap(frame=frame, prediction=prediction)
    record = DecisionEngine().decide(frame=frame, prediction=prediction)
    record.heatmap_path = heatmap_path
    writer.append_decision(record)

    assert heatmap_path is not None
    assert (tmp_path / heatmap_path).exists()
    assert writer.predictions_path.exists()


def test_runtime_app_emits_decision_for_every_frame(tmp_path: Path, monkeypatch):
    frames = [
        _frame(np.zeros((12, 12, 3), dtype=np.uint8), frame_id=1),
        _frame(np.full((12, 12, 3), 255, dtype=np.uint8), frame_id=2),
    ]

    class _FakeInputHandler:
        def __init__(self, **_: object) -> None:
            pass

        def iter_frames(self):
            yield from frames

    class _FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, _: np.ndarray):
            self.calls += 1
            is_anomaly = 1 if self.calls == 2 else 0
            heatmap = np.ones((6, 6), dtype=np.float32) if is_anomaly else None
            return SimpleNamespace(score=0.2 * self.calls, is_anomaly=is_anomaly, heatmap=heatmap)

    class _FakeRegistry:
        def __init__(self, artifact: object, fit_policy: str = "auto") -> None:
            self.artifact = artifact
            self.fit_policy = fit_policy

        def load(self) -> _FakeModel:
            return _FakeModel()

    artifact = SimpleNamespace(model_name="stub_model", threshold=0.42)

    monkeypatch.setattr(app_module, "load_runtime_artifact", lambda **_: artifact)
    monkeypatch.setattr(app_module, "SingleModelRegistry", _FakeRegistry)
    monkeypatch.setattr(app_module, "FolderInputHandler", _FakeInputHandler)

    cfg = {
        "run": {
            "output_dir": str(tmp_path),
            "session_name": "inspection_test",
            "target_fps": 0.0,
            "max_frames": 2,
        },
        "artifact": {
            "run_dir": str(tmp_path),
            "model_name": "stub_model",
        },
        "input": {
            "root_dir": str(tmp_path),
            "loop": False,
            "sequence_mode": "interleaved_labels",
        },
        "preprocessing": {
            "resize": {"enabled": False},
            "normalize": {"enabled": True},
        },
        "web": {"enabled": False},
    }

    session_dir = app_module.StreamingInputApp(cfg).run()

    predictions = json.loads((session_dir / "predictions.json").read_text(encoding="utf-8"))
    status = json.loads((session_dir / "live_status.json").read_text(encoding="utf-8"))

    assert len(predictions) == 2
    assert [row["path"] for row in predictions] == ["sample.png", "sample.png"]
    assert status["decisions_emitted"] == 2
    assert status["frames_seen"] == 2
    assert status["fail_count"] == 1
    assert status["active_model"] == "stub_model"
    assert status["threshold"] == 0.42
    assert len(status["recent_fails"]) == 1
    assert (session_dir / "heatmaps" / "frame_000002.png").exists()
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


def test_interleave_by_label_mixes_good_bad_and_unlabeled():
    samples = [
        LabeledSample(path=Path("good_1.png"), label=0),
        LabeledSample(path=Path("good_2.png"), label=0),
        LabeledSample(path=Path("bad_1.png"), label=1),
        LabeledSample(path=Path("bad_2.png"), label=1),
        LabeledSample(path=Path("unknown_1.png"), label=-1),
    ]

    ordered = _interleave_by_label(samples)
    ordered_paths = [sample.path.name for sample in ordered]
    assert ordered_paths == [
        "good_1.png",
        "bad_1.png",
        "unknown_1.png",
        "good_2.png",
        "bad_2.png",
    ]

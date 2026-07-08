from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .scorer import TruthRow, score_round
from .types import Detection, Frame, Instance, OutputRow


def detection_to_dict(det: Detection) -> dict[str, Any]:
    return {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "adjusted_confidence": det.adjusted_confidence,
        "xyxy": list(det.xyxy),
        "frame_index": det.frame_index,
        "table_id": det.table_id,
        "center_3d": None if det.center_3d is None else list(det.center_3d),
        "depth_score": det.depth_score,
        "picture_like": det.picture_like,
        "depth_stats": det.depth_stats,
    }


def instance_to_dict(inst: Instance) -> dict[str, Any]:
    return {
        "class_name": inst.class_name,
        "table_id": inst.table_id,
        "score": inst.score,
        "visible_frames": inst.visible_frames,
        "best_xyxy": list(inst.best_xyxy),
        "center_3d": None if inst.center_3d is None else list(inst.center_3d),
        "class_scores": inst.class_scores,
        "classifier_scores": inst.classifier_scores,
        "classifier_label": inst.classifier_label,
        "classifier_score": inst.classifier_score,
        "depth_validated": inst.depth_validated,
        "picture_like_votes": inst.picture_like_votes,
        "ocr_text": inst.ocr_text,
        "ocr_score": inst.ocr_score,
        "diagnostics": inst.diagnostics,
    }


def output_to_dict(row: OutputRow) -> dict[str, Any]:
    return asdict(row)


def parse_truth_file(path: str | Path | None) -> list[TruthRow]:
    if not path:
        return []
    truth_path = Path(path)
    if not truth_path.exists():
        raise FileNotFoundError(truth_path)
    if truth_path.suffix.lower() == ".json":
        raw = json.loads(truth_path.read_text(encoding="utf-8"))
        rows = raw.get("truth", raw) if isinstance(raw, dict) else raw
        return [
            TruthRow(
                object_id=str(item.get("object_id", item.get("class_name"))),
                num=int(item.get("num", 1)),
                table_id=int(item.get("table_id", 1)),
            )
            for item in rows
        ]

    rows: list[TruthRow] = []
    for line in truth_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line in {"START", "END"}:
            continue
        parts = [part.strip() for part in line.replace(",", ";").split(";")]
        if len(parts) < 3:
            continue
        rows.append(TruthRow(object_id=parts[0], num=int(parts[1]), table_id=int(parts[2])))
    return rows


class ReplayRecorder:
    def __init__(self, session_dir: str | Path | None) -> None:
        self.session_dir = Path(session_dir) if session_dir else None
        self.frames_dir = None if self.session_dir is None else self.session_dir / "frames"
        self.detections: list[dict[str, Any]] = []
        if self.session_dir is not None:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.session_dir is not None

    def record_frame(self, frame: Frame, detections: list[Detection]) -> None:
        if not self.enabled or self.frames_dir is None:
            return
        import cv2

        cv2.imwrite(str(self.frames_dir / f"rgb_{frame.index:04d}.jpg"), frame.rgb)
        if frame.depth is not None:
            np.save(self.frames_dir / f"depth_{frame.index:04d}.npy", frame.depth)
        self.detections.append(
            {
                "frame_index": frame.index,
                "timestamp": frame.timestamp,
                "detections": [detection_to_dict(det) for det in detections],
            }
        )

    def write_final(
        self,
        instances: list[Instance],
        rows: list[OutputRow],
        elapsed_sec: float,
        truth_path: str | Path | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled or self.session_dir is None:
            return None

        truth = parse_truth_file(truth_path)
        score = score_round(truth, rows) if truth else None
        payload = {
            "elapsed_sec": elapsed_sec,
            "detections": self.detections,
            "instances": [instance_to_dict(inst) for inst in instances],
            "outputs": [output_to_dict(row) for row in rows],
            "truth": [asdict(row) for row in truth],
            "score": None if score is None else asdict(score),
        }
        (self.session_dir / "run_record.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        validation_sample = {
            "instances": payload["instances"],
            "truth": payload["truth"],
        }
        (self.session_dir / "validation_sample.json").write_text(
            json.dumps({"samples": [validation_sample]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload


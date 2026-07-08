from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import RuntimeConfig
from .decision import decide_outputs
from .scorer import TruthRow, score_round
from .types import Instance


def _instance_from_dict(data: dict[str, Any]) -> Instance:
    return Instance(
        class_name=str(data["class_name"]),
        table_id=int(data.get("table_id", 1)),
        score=float(data.get("score", data.get("confidence", 0.0))),
        visible_frames=int(data.get("visible_frames", 1)),
        best_xyxy=tuple(float(v) for v in data.get("best_xyxy", [0, 0, 1, 1])),
        depth_validated=bool(data.get("depth_validated", False)),
        picture_like_votes=int(data.get("picture_like_votes", 0)),
    )


def _truth_from_dict(data: dict[str, Any]) -> TruthRow:
    return TruthRow(
        object_id=str(data.get("object_id", data.get("class_name"))),
        num=int(data.get("num", 1)),
        table_id=int(data.get("table_id", 1)),
    )


def load_validation_samples(path: str | Path) -> list[tuple[list[Instance], list[TruthRow]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
    parsed: list[tuple[list[Instance], list[TruthRow]]] = []
    for sample in samples:
        instances = [_instance_from_dict(item) for item in sample.get("instances", [])]
        truth = [_truth_from_dict(item) for item in sample.get("truth", [])]
        parsed.append((instances, truth))
    return parsed


def total_measure_score(samples: list[tuple[list[Instance], list[TruthRow]]], runtime: RuntimeConfig) -> float:
    total = 0.0
    for instances, truth in samples:
        prediction = decide_outputs(instances, runtime)
        total += score_round(truth, prediction).measure_score
    return total


def optimize_class_thresholds(
    samples: list[tuple[list[Instance], list[TruthRow]]],
    runtime: RuntimeConfig,
    candidates: list[float] | None = None,
) -> dict[str, float]:
    candidates = candidates or [round(x / 100.0, 2) for x in range(10, 96, 5)]
    classes = sorted({inst.class_name for instances, _ in samples for inst in instances})
    optimized = dict(runtime.class_thresholds)
    working = deepcopy(runtime)

    for class_name in classes:
        best_threshold = optimized.get(class_name, runtime.conf_floor)
        best_score = None
        for threshold in candidates:
            working.class_thresholds = {**optimized, class_name: threshold}
            score = total_measure_score(samples, working)
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = threshold
        optimized[class_name] = float(best_threshold)

    return optimized


def write_threshold_yaml(path: str | Path, thresholds: dict[str, float]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            {"runtime": {"class_thresholds": thresholds}},
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


from __future__ import annotations

from dataclasses import dataclass

from .types import OutputRow


@dataclass(slots=True)
class TruthRow:
    object_id: str
    num: int
    table_id: int


@dataclass(slots=True)
class ScoreResult:
    measure_score: float
    full_score: float
    average_score: float
    false_id_penalty: float


def score_round(truth: list[TruthRow], prediction: list[OutputRow]) -> ScoreResult:
    truth_map = {(row.object_id, row.table_id): row.num for row in truth}
    pred_map = {(row.object_id, row.table_id): row.num for row in prediction}

    measure = 0.0
    false_penalty = 0.0

    for key, pred_num in pred_map.items():
        true_num = truth_map.get(key)
        if true_num is None:
            measure -= 3.0
            false_penalty -= 3.0
            continue
        if pred_num == true_num:
            measure += 3.0
        elif pred_num < true_num:
            measure += pred_num / true_num * 3.0

    full_score = 3.0 * len(truth_map)
    avg = 0.0 if len(truth_map) == 0 else measure / len(truth_map)
    return ScoreResult(
        measure_score=measure,
        full_score=full_score,
        average_score=avg,
        false_id_penalty=false_penalty,
    )


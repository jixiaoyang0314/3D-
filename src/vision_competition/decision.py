from __future__ import annotations

from collections import defaultdict

from .config import RuntimeConfig
from .types import Instance, OutputRow


def poisson_binomial_distribution(probabilities: list[float], max_n: int) -> list[float]:
    dist = [1.0] + [0.0] * max_n
    for p in probabilities:
        p = max(0.0, min(1.0, p))
        next_dist = [0.0] * (max_n + 1)
        for n in range(max_n + 1):
            next_dist[n] += dist[n] * (1.0 - p)
            if n + 1 <= max_n:
                next_dist[n + 1] += dist[n] * p
        dist = next_dist
    return dist


def expected_competition_score(output_num: int, true_num: int) -> float:
    if output_num <= 0:
        return 0.0
    if true_num <= 0:
        return -3.0
    if output_num == true_num:
        return 3.0
    if output_num < true_num:
        return output_num / true_num * 3.0
    return 0.0


def class_threshold(class_name: str, runtime: RuntimeConfig) -> float:
    return float(runtime.class_thresholds.get(class_name, runtime.conf_floor))


def class_min_expected_score(class_name: str, runtime: RuntimeConfig) -> float:
    return float(runtime.class_min_expected_scores.get(class_name, runtime.output_min_expected_score))


def choose_count(
    probabilities: list[float],
    runtime: RuntimeConfig,
    min_expected_score: float | None = None,
) -> tuple[int, float]:
    if not probabilities:
        return 0, 0.0

    probabilities = sorted((max(0.0, min(1.0, p)) for p in probabilities), reverse=True)
    probabilities = probabilities[: runtime.max_count_per_id]
    max_n = runtime.max_count_per_id
    dist = poisson_binomial_distribution(probabilities, max_n=max_n)

    best_k = 0
    best_expected = 0.0
    for k in range(1, min(len(probabilities), runtime.max_count_per_id) + 1):
        expected = sum(dist[n] * expected_competition_score(k, n) for n in range(max_n + 1))
        if expected > best_expected:
            best_expected = expected
            best_k = k

    threshold = runtime.output_min_expected_score if min_expected_score is None else min_expected_score
    if best_expected < threshold:
        return 0, best_expected
    return best_k, best_expected


def decide_outputs(instances: list[Instance], runtime: RuntimeConfig) -> list[OutputRow]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)

    for inst in instances:
        if inst.score < class_threshold(inst.class_name, runtime):
            continue
        if inst.visible_frames < runtime.stable_min_frames and inst.score < runtime.high_conf:
            continue
        if inst.picture_like_votes >= max(1, inst.visible_frames // 2 + 1):
            continue
        grouped[(inst.class_name, inst.table_id)].append(inst.score)

    rows: list[OutputRow] = []
    for (class_name, table_id), probs in grouped.items():
        num, expected = choose_count(
            probs,
            runtime,
            min_expected_score=class_min_expected_score(class_name, runtime),
        )
        if num > 0:
            rows.append(OutputRow(object_id=class_name, num=num, table_id=table_id, score=expected))

    rows.sort(key=lambda row: (row.table_id, row.object_id))
    return rows

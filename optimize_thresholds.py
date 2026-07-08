from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_competition.config import load_config
from vision_competition.threshold_optimizer import (
    load_validation_samples,
    optimize_class_thresholds,
    total_measure_score,
    write_threshold_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn per-class output thresholds from validation replays.")
    parser.add_argument("--config", default="configs/competition.yaml")
    parser.add_argument("--validation", required=True, help="JSON file with instances and truth rows.")
    parser.add_argument("--output", default="configs/learned_thresholds.yaml")
    parser.add_argument("--grid", default="0.10:0.95:0.05", help="start:end:step threshold grid.")
    return parser.parse_args()


def parse_grid(spec: str) -> list[float]:
    start, end, step = (float(part) for part in spec.split(":"))
    values: list[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    samples = load_validation_samples(args.validation)
    before = total_measure_score(samples, config.runtime)
    thresholds = optimize_class_thresholds(samples, config.runtime, candidates=parse_grid(args.grid))
    config.runtime.class_thresholds = thresholds
    after = total_measure_score(samples, config.runtime)
    output = write_threshold_yaml(args.output, thresholds)
    print(f"samples={len(samples)} before={before:.3f} after={after:.3f}")
    print(f"written={output}")


if __name__ == "__main__":
    main()


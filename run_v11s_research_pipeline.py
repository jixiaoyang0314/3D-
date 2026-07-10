from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
RUN_ROOT = ROOT / "runs" / "detect" / "runs" / "train"
LOG_ROOT = ROOT / "runs" / "logs"
STATE_PATH = ROOT / "runs" / "v11s_research_pipeline_state.json"
SUMMARY_PATH = ROOT / "runs" / "v11s_research_pipeline_summary.json"

STAGE1_NAME = "v11s_640_moderate_stage1"
QUIET_NAME = "v11s_640_moderate_quiet_finetune"
DISTILL_SEEDS = (20260708, 20260709, 20260710)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete the YOLO11s precision research pipeline.")
    parser.add_argument("--stage1-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(stage: str, status: str, **details: object) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now_iso(),
        "stage": stage,
        "status": status,
        **details,
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_matches(pid: int, expected: str) -> bool:
    if not psutil.pid_exists(pid):
        return False
    try:
        command = " ".join(psutil.Process(pid).cmdline())
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    return expected in command


def run_dir(name: str) -> Path:
    return RUN_ROOT / name


def require_best(name: str) -> Path:
    path = run_dir(name) / "weights" / "best.pt"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing best checkpoint for {name}: {path}")
    return path


def best_training_row(name: str) -> dict[str, str]:
    results_path = run_dir(name) / "results.csv"
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No training metrics found for {name}")
    return max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))


def run_logged(stage: str, command: list[str]) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = LOG_ROOT / f"{stage}.out.log"
    err_path = LOG_ROOT / f"{stage}.err.log"
    write_state(stage, "running", command=command, stdout=str(out_path), stderr=str(err_path))
    with out_path.open("w", encoding="utf-8", errors="replace") as stdout, err_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    if result.returncode != 0:
        write_state(stage, "failed", returncode=result.returncode, stderr=str(err_path))
        raise RuntimeError(f"{stage} failed with exit code {result.returncode}; see {err_path}")
    write_state(stage, "completed", returncode=0)


def common_quiet_args(data: Path, weights: Path, name: str, seed: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "train_yolo.py"),
        "--data",
        str(data),
        "--weights",
        str(weights),
        "--preset",
        "fine_tune",
        "--epochs",
        "30",
        "--imgsz",
        "640",
        "--batch",
        "16",
        "--device",
        "0",
        "--workers",
        "4",
        "--optimizer",
        "AdamW",
        "--lr0",
        "0.00008",
        "--lrf",
        "0.1",
        "--weight-decay",
        "0.0005",
        "--warmup-epochs",
        "1",
        "--mosaic",
        "0",
        "--mixup",
        "0",
        "--copy-paste",
        "0",
        "--erasing",
        "0",
        "--degrees",
        "2",
        "--translate",
        "0.02",
        "--scale",
        "0.08",
        "--shear",
        "0",
        "--perspective",
        "0",
        "--hsv-h",
        "0.003",
        "--hsv-s",
        "0.08",
        "--hsv-v",
        "0.08",
        "--fliplr",
        "0",
        "--box",
        "7.5",
        "--cls",
        "0.5",
        "--dfl",
        "1.5",
        "--cls-pw",
        "0.2",
        "--patience",
        "10",
        "--seed",
        str(seed),
        "--project",
        "runs/train",
        "--name",
        name,
    ]


def make_light_dataset() -> Path:
    output = WORKSPACE / "yolo_dataset_official_light_aug"
    data_yaml = output / "data.yaml"
    if data_yaml.exists():
        return data_yaml
    command = [
        sys.executable,
        str(ROOT / "augment_official_dataset.py"),
        "--data",
        str(WORKSPACE / "yolo_dataset_official" / "data.yaml"),
        "--output",
        str(output),
        "--yellow",
        "0.10",
        "--white",
        "0.10",
        "--dark",
        "0.05",
        "--balance-target",
        "none",
        "--copy-originals",
        "reencode",
        "--max-side",
        "1920",
        "--quality",
        "95",
    ]
    run_logged("v11s_prepare_light_aug", command)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Light augmentation dataset was not created: {data_yaml}")
    return data_yaml


def run_quiet_finetune(data_yaml: Path, stage1_best: Path) -> Path:
    quiet_best = run_dir(QUIET_NAME) / "weights" / "best.pt"
    if not quiet_best.exists():
        run_logged(QUIET_NAME, common_quiet_args(data_yaml, stage1_best, QUIET_NAME, DISTILL_SEEDS[0]))
    return require_best(QUIET_NAME)


def run_distillations(data_yaml: Path, student_best: Path, teacher_best: Path) -> list[Path]:
    outputs: list[Path] = []
    for seed in DISTILL_SEEDS:
        name = f"v11s_640_distill_seed{seed}"
        best = run_dir(name) / "weights" / "best.pt"
        if not best.exists():
            command = common_quiet_args(data_yaml, student_best, name, seed)
            command[command.index("--epochs") + 1] = "24"
            command[command.index("--batch") + 1] = "8"
            command[command.index("--lr0") + 1] = "0.00005"
            command[command.index("--lrf") + 1] = "0.2"
            command[command.index("--patience") + 1] = "8"
            command.extend(["--distill-model", str(teacher_best), "--distill-weight", "2.0"])
            run_logged(name, command)
        outputs.append(require_best(name))
    return outputs


def validate_candidates(candidates: dict[str, Path]) -> list[dict[str, object]]:
    from ultralytics import YOLO

    from train_yolo import resolve_data_yaml

    data = resolve_data_yaml(str(WORKSPACE / "yolo_dataset_official" / "data.yaml"))
    metrics_rows: list[dict[str, object]] = []
    for name, weights in candidates.items():
        write_state("validation_704", "running", candidate=name, weights=str(weights))
        model = YOLO(str(weights))
        metrics = model.val(data=data, imgsz=704, batch=16, device=0, workers=4, plots=False, verbose=False)
        metrics_rows.append(
            {
                "name": name,
                "weights": str(weights),
                "map50_95": float(metrics.box.map),
                "map50": float(metrics.box.map50),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "per_class_map50_95": {
                    str(model.names[index]): float(value) for index, value in enumerate(metrics.box.maps)
                },
            }
        )
    return sorted(metrics_rows, key=lambda row: float(row["map50_95"]), reverse=True)


def main() -> None:
    args = parse_args()
    write_state(STAGE1_NAME, "waiting", pid=args.stage1_pid)
    while process_matches(args.stage1_pid, STAGE1_NAME):
        time.sleep(max(5, args.poll_seconds))

    stage1_best = require_best(STAGE1_NAME)
    stage1_row = best_training_row(STAGE1_NAME)
    write_state(
        STAGE1_NAME,
        "completed",
        best_epoch=int(float(stage1_row["epoch"])),
        best_map50_95=float(stage1_row["metrics/mAP50-95(B)"]),
        weights=str(stage1_best),
    )

    data_yaml = make_light_dataset()
    quiet_best = run_quiet_finetune(data_yaml, stage1_best)
    teacher_best = require_best("v11m_640_quiet_finetune")
    distill_best = run_distillations(data_yaml, quiet_best, teacher_best)

    candidates = {
        "v11s_existing_quiet_baseline": require_best("v11s_640_quiet_finetune"),
        QUIET_NAME: quiet_best,
    }
    candidates.update({f"v11s_640_distill_seed{seed}": path for seed, path in zip(DISTILL_SEEDS, distill_best)})
    validation = validate_candidates(candidates)
    selected = validation[0]
    destination = ROOT / "weights" / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(str(selected["weights"])), destination)

    summary = {
        "completed_at": now_iso(),
        "selection_metric": "mAP50-95 at imgsz=704 on yolo_dataset_official/val",
        "selected": selected,
        "deployed_weights": str(destination),
        "stage1_training_best": {
            "epoch": int(float(stage1_row["epoch"])),
            "map50_95": float(stage1_row["metrics/mAP50-95(B)"]),
        },
        "candidates": validation,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_state("pipeline", "completed", selected=selected, summary=str(SUMMARY_PATH))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_state("pipeline", "failed", error=repr(exc))
        raise

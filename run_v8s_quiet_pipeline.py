from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DATA = WORKSPACE / "yolo_dataset_official_aug" / "data.yaml"
STATE_PATH = ROOT / "runs" / "v8s_quiet_pipeline_state.json"
SUMMARY_PATH = ROOT / "runs" / "v8s_quiet_pipeline_summary.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(stage: str, status: str, **details: object) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": now_iso(), "stage": stage, "status": status, **details}
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_dir(name: str) -> Path:
    return ROOT / "runs" / "detect" / "runs" / "train" / name


def best_path(name: str) -> Path:
    path = run_dir(name) / "weights" / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing best checkpoint for {name}: {path}")
    return path


def best_training_row(name: str) -> dict[str, str]:
    path = run_dir(name) / "results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))


def run_logged(stage: str, command: list[str]) -> None:
    logs = ROOT / "runs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{stage}.out.log"
    stderr_path = logs / f"{stage}.err.log"
    write_state(stage, "running", command=command, stdout=str(stdout_path), stderr=str(stderr_path))
    with stdout_path.open("w", encoding="utf-8", errors="replace") as out, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as err:
        subprocess.run(command, cwd=ROOT, check=True, stdout=out, stderr=err)
    write_state(stage, "completed")


def train_stage(stage: str, weights: str | Path, extra: list[str]) -> Path:
    target = best_path(stage) if (run_dir(stage) / "weights" / "best.pt").exists() else None
    if target is not None:
        return target
    command = [sys.executable, str(ROOT / "train_yolo.py"), "--data", str(DATA), "--weights", str(weights), *extra]
    run_logged(stage, command)
    return best_path(stage)


def validate_704(candidates: dict[str, Path]) -> list[dict[str, object]]:
    from ultralytics import YOLO

    from train_yolo import resolve_data_yaml

    data = resolve_data_yaml(str(WORKSPACE / "yolo_dataset_official" / "data.yaml"))
    rows: list[dict[str, object]] = []
    for name, weights in candidates.items():
        write_state("validation_704", "running", candidate=name, weights=str(weights))
        model = YOLO(str(weights))
        metrics = model.val(data=data, imgsz=704, batch=16, device=0, workers=4, plots=False, verbose=False)
        rows.append(
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
    return sorted(rows, key=lambda row: float(row["map50_95"]), reverse=True)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    # Same chain as v11s_640_quiet_finetune, replacing only YOLO11s with YOLOv8s.
    stage1 = train_stage(
        "v8s_640_stage1_balanced",
        "yolov8s.pt",
        [
            "--preset",
            "balanced",
            "--epochs",
            "120",
            "--imgsz",
            "640",
            "--batch",
            "8",
            "--device",
            "0",
            "--workers",
            "4",
            "--optimizer",
            "auto",
            "--patience",
            "50",
            "--project",
            "runs/train",
            "--name",
            "v8s_640_stage1_balanced",
        ],
    )
    finetune = train_stage(
        "v8s_640_finetune",
        stage1,
        [
            "--preset",
            "fine_tune",
            "--epochs",
            "80",
            "--imgsz",
            "640",
            "--batch",
            "8",
            "--device",
            "0",
            "--workers",
            "4",
            "--optimizer",
            "auto",
            "--lr0",
            "0.001",
            "--lrf",
            "0.01",
            "--warmup-epochs",
            "3",
            "--patience",
            "30",
            "--project",
            "runs/train",
            "--name",
            "v8s_640_finetune",
        ],
    )
    quiet = train_stage(
        "v8s_640_quiet_finetune",
        finetune,
        [
            "--preset",
            "fine_tune",
            "--epochs",
            "50",
            "--imgsz",
            "640",
            "--batch",
            "8",
            "--device",
            "0",
            "--workers",
            "4",
            "--optimizer",
            "AdamW",
            "--lr0",
            "0.00015",
            "--lrf",
            "0.1",
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
            "--fliplr",
            "0",
            "--box",
            "7.5",
            "--cls",
            "0.5",
            "--dfl",
            "1.5",
            "--patience",
            "15",
            "--project",
            "runs/train",
            "--name",
            "v8s_640_quiet_finetune",
        ],
    )

    candidates = {
        "v11s_640_quiet_finetune": best_path("v11s_640_quiet_finetune"),
        "v8s_640_quiet_finetune": quiet,
    }
    validation = validate_704(candidates)
    summary = {
        "completed_at": now_iso(),
        "selection_metric": "mAP50-95 at imgsz=704 on yolo_dataset_official/val",
        "selected": validation[0],
        "v8_stage1_training_best": {
            "epoch": int(float(best_training_row("v8s_640_stage1_balanced")["epoch"])),
            "map50_95": float(best_training_row("v8s_640_stage1_balanced")["metrics/mAP50-95(B)"]),
        },
        "v8_finetune_training_best": {
            "epoch": int(float(best_training_row("v8s_640_finetune")["epoch"])),
            "map50_95": float(best_training_row("v8s_640_finetune")["metrics/mAP50-95(B)"]),
        },
        "v8_quiet_training_best": {
            "epoch": int(float(best_training_row("v8s_640_quiet_finetune")["epoch"])),
            "map50_95": float(best_training_row("v8s_640_quiet_finetune")["metrics/mAP50-95(B)"]),
        },
        "candidates": validation,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if validation[0]["name"] == "v8s_640_quiet_finetune":
        destination = ROOT / "weights" / "best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(quiet, destination)
        summary["deployed_weights"] = str(destination)
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_state("pipeline", "completed", summary=str(SUMMARY_PATH), selected=validation[0])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_state("pipeline", "failed", error=repr(exc))
        raise

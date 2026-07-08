from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for the 3D recognition contest.")
    parser.add_argument("--data", required=True, help="Ultralytics data yaml.")
    parser.add_argument("--weights", default="yolo11s.pt", help="Initial YOLO weights.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="contest_yolo")
    parser.add_argument("--task", choices=["detect", "segment"], default="detect")
    return parser.parse_args()


def resolve_data_yaml(data: str) -> str:
    data_path = Path(data)
    if not data_path.exists() or data_path.suffix.lower() not in {".yaml", ".yml"}:
        return data

    raw = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return data

    root = Path(raw.get("path") or data_path.parent)
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    raw["path"] = root.as_posix()

    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix="robot_yolo_",
        delete=False,
    )
    with temp:
        yaml.safe_dump(raw, temp, allow_unicode=True, sort_keys=False)
    return temp.name


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics first: pip install ultralytics") from exc

    model = YOLO(args.weights, task=args.task)
    model.train(
        data=resolve_data_yaml(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=40,
        close_mosaic=20,
        cos_lr=True,
        hsv_h=0.03,
        hsv_s=0.65,
        hsv_v=0.45,
        degrees=25.0,
        translate=0.15,
        scale=0.55,
        shear=4.0,
        perspective=0.0008,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.08,
        copy_paste=0.15,
    )


if __name__ == "__main__":
    main()

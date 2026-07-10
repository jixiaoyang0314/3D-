from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml


TRAINING_PRESETS = {
    "balanced": {
        "patience": 50,
        "workers": 8,
        "seed": 20260708,
        "mosaic": 0.8,
        "close_mosaic": 25,
        "mixup": 0.08,
        "copy_paste": 0.15,
        "hsv_h": 0.03,
        "hsv_s": 0.65,
        "hsv_v": 0.45,
        "degrees": 25.0,
        "translate": 0.15,
        "scale": 0.55,
        "shear": 4.0,
        "perspective": 0.0008,
        "fliplr": 0.5,
        "multi_scale": 0.0,
    },
    "moderate": {
        "patience": 35,
        "workers": 8,
        "seed": 20260708,
        "mosaic": 0.5,
        "close_mosaic": 20,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.35,
        "hsv_v": 0.25,
        "degrees": 15.0,
        "translate": 0.10,
        "scale": 0.35,
        "shear": 2.0,
        "perspective": 0.0003,
        "fliplr": 0.5,
        "multi_scale": 0.10,
    },
    "precision": {
        "patience": 60,
        "workers": 8,
        "seed": 20260708,
        "mosaic": 0.35,
        "close_mosaic": 40,
        "mixup": 0.0,
        "copy_paste": 0.05,
        "hsv_h": 0.015,
        "hsv_s": 0.45,
        "hsv_v": 0.30,
        "degrees": 12.0,
        "translate": 0.10,
        "scale": 0.35,
        "shear": 2.0,
        "perspective": 0.0003,
        "fliplr": 0.5,
        "multi_scale": 0.0,
    },
    "fine_tune": {
        "patience": 30,
        "workers": 8,
        "seed": 20260708,
        "mosaic": 0.05,
        "close_mosaic": 0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.01,
        "hsv_s": 0.25,
        "hsv_v": 0.18,
        "degrees": 6.0,
        "translate": 0.05,
        "scale": 0.20,
        "shear": 1.0,
        "perspective": 0.0001,
        "fliplr": 0.3,
        "multi_scale": 0.0,
    },
}


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
    parser.add_argument(
        "--preset",
        choices=sorted(TRAINING_PRESETS),
        default="balanced",
        help="Training augmentation preset. Explicit CLI values override preset defaults.",
    )
    parser.add_argument("--patience", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, help="Initial learning rate. Leave unset for Ultralytics auto/default.")
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--box", type=float, help="Box loss gain.")
    parser.add_argument("--cls", type=float, help="Class loss gain.")
    parser.add_argument("--dfl", type=float, help="DFL loss gain.")
    parser.add_argument("--cls-pw", type=float, help="Inverse-frequency class weighting power.")
    parser.add_argument("--multi-scale", type=float, help="Per-batch image-size variation fraction.")
    parser.add_argument("--distill-model", help="Teacher checkpoint used for knowledge distillation.")
    parser.add_argument("--distill-weight", dest="dis", type=float, help="Knowledge-distillation loss weight.")
    parser.add_argument("--rect", action="store_true", help="Use rectangular training batches.")
    parser.add_argument("--cache", choices=["ram", "disk"], help="Cache images for faster training.")
    parser.add_argument("--mosaic", type=float)
    parser.add_argument("--close-mosaic", type=int)
    parser.add_argument("--mixup", type=float)
    parser.add_argument("--copy-paste", type=float)
    parser.add_argument("--hsv-h", type=float)
    parser.add_argument("--hsv-s", type=float)
    parser.add_argument("--hsv-v", type=float)
    parser.add_argument("--degrees", type=float)
    parser.add_argument("--translate", type=float)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--shear", type=float)
    parser.add_argument("--perspective", type=float)
    parser.add_argument("--fliplr", type=float)
    parser.add_argument("--erasing", type=float)
    args = parser.parse_args()
    apply_preset_defaults(args)
    return args


def apply_preset_defaults(args: argparse.Namespace) -> None:
    for key, value in TRAINING_PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)


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
    train_kwargs = {
        "data": resolve_data_yaml(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "fraction": args.fraction,
        "optimizer": args.optimizer,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "multi_scale": args.multi_scale,
        "rect": args.rect,
        "close_mosaic": args.close_mosaic,
        "cos_lr": True,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
    }
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.cache is not None:
        train_kwargs["cache"] = args.cache
    if args.erasing is not None:
        train_kwargs["erasing"] = args.erasing
    if args.box is not None:
        train_kwargs["box"] = args.box
    if args.cls is not None:
        train_kwargs["cls"] = args.cls
    if args.dfl is not None:
        train_kwargs["dfl"] = args.dfl
    if args.cls_pw is not None:
        train_kwargs["cls_pw"] = args.cls_pw
    if args.distill_model is not None:
        train_kwargs["distill_model"] = args.distill_model
    if args.dis is not None:
        train_kwargs["dis"] = args.dis

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

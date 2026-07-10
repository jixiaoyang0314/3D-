from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageFile, ImageOps


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CROP_CLASSES = (0, 3, 4, 6, 8)


@dataclass(frozen=True)
class YoloBox:
    cls: int
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Candidate:
    image: Path
    label: Path
    boxes: tuple[YoloBox, ...]
    target_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a YOLO dataset with crop augmentations for small or weak objects."
    )
    parser.add_argument("--data", required=True, help="Source YOLO data.yaml.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    parser.add_argument("--crop-classes", default=",".join(map(str, DEFAULT_CROP_CLASSES)))
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=0.08,
        help="Only crop target boxes with normalized area at or below this value.",
    )
    parser.add_argument("--max-crops-per-class", type=int, default=600)
    parser.add_argument("--crop-scale", type=float, default=4.0, help="Crop side multiplier around target size.")
    parser.add_argument("--min-crop-side", type=int, default=224)
    parser.add_argument("--jitter", type=float, default=0.15, help="Random center jitter as a fraction of crop side.")
    parser.add_argument("--min-visibility", type=float, default=0.35)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-originals", choices=["copy", "hardlink"], default="hardlink")
    return parser.parse_args()


def load_data_yaml(path: Path) -> tuple[dict, Path]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a mapping-style YOLO data yaml")
    root = Path(raw.get("path") or path.parent)
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return raw, root


def resolve_split(root: Path, split_value: str | Path) -> list[Path]:
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path

    if split_path.is_file():
        images: list[Path] = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            image_path = Path(line)
            images.append(image_path if image_path.is_absolute() else root / image_path)
        return sorted(images)

    return sorted(path for path in split_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def image_to_label(image_path: Path, root: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return root / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def read_boxes(label_path: Path) -> tuple[YoloBox, ...]:
    boxes: list[YoloBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = line.split()[:5]
        boxes.append(YoloBox(int(float(cls)), float(x), float(y), float(w), float(h)))
    return tuple(boxes)


def prepare_output(output: Path, source_root: Path, overwrite: bool) -> None:
    output = output.resolve()
    source_root = source_root.resolve()
    if output == source_root:
        raise ValueError("Refusing to overwrite the source dataset.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def place_file(source: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            dest.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, dest)


def copy_split(images: list[Path], root: Path, output: Path, split: str, mode: str) -> None:
    used_stems: Counter[str] = Counter()
    for image in images:
        label = image_to_label(image, root)
        if not label.exists():
            raise FileNotFoundError(f"Missing label for {image}: {label}")
        used_stems[image.stem] += 1
        stem = image.stem if used_stems[image.stem] == 1 else f"{image.stem}__dup_{used_stems[image.stem]}"
        out_image = output / "images" / split / f"{stem}{image.suffix.lower()}"
        out_label = output / "labels" / split / f"{stem}.txt"
        place_file(image, out_image, mode)
        shutil.copy2(label, out_label)


def yolo_to_xyxy(box: YoloBox, width: int, height: int) -> tuple[float, float, float, float]:
    cx = box.x * width
    cy = box.y * height
    bw = box.w * width
    bh = box.h * height
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def make_crop_window(
    target: YoloBox,
    width: int,
    height: int,
    crop_scale: float,
    min_crop_side: int,
    jitter: float,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = yolo_to_xyxy(target, width, height)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    side = max(min_crop_side, round(max(box_w, box_h) * crop_scale))
    crop_w = min(width, side)
    crop_h = min(height, side)
    cx = (x1 + x2) / 2 + rng.uniform(-jitter, jitter) * crop_w
    cy = (y1 + y2) / 2 + rng.uniform(-jitter, jitter) * crop_h
    left = round(clamp(cx - crop_w / 2, 0, width - crop_w))
    top = round(clamp(cy - crop_h / 2, 0, height - crop_h))
    return left, top, left + crop_w, top + crop_h


def crop_labels(
    boxes: tuple[YoloBox, ...],
    width: int,
    height: int,
    crop: tuple[int, int, int, int],
    min_visibility: float,
) -> list[YoloBox]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    cropped: list[YoloBox] = []

    for box in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(box, width, height)
        inter_x1 = max(x1, crop_x1)
        inter_y1 = max(y1, crop_y1)
        inter_x2 = min(x2, crop_x2)
        inter_y2 = min(y2, crop_y2)
        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1
        if inter_w <= 1 or inter_h <= 1:
            continue
        original_area = max(1.0, (x2 - x1) * (y2 - y1))
        if (inter_w * inter_h) / original_area < min_visibility:
            continue
        new_x = ((inter_x1 + inter_x2) / 2 - crop_x1) / crop_w
        new_y = ((inter_y1 + inter_y2) / 2 - crop_y1) / crop_h
        new_w = inter_w / crop_w
        new_h = inter_h / crop_h
        cropped.append(YoloBox(box.cls, new_x, new_y, new_w, new_h))

    return cropped


def save_boxes(boxes: list[YoloBox], path: Path) -> None:
    lines = [f"{box.cls} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}" for box in boxes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_candidates(
    images: list[Path],
    root: Path,
    crop_classes: set[int],
    area_threshold: float,
) -> dict[int, list[Candidate]]:
    candidates: dict[int, list[Candidate]] = defaultdict(list)
    for image in images:
        label = image_to_label(image, root)
        boxes = read_boxes(label)
        for index, box in enumerate(boxes):
            if box.cls in crop_classes and box.w * box.h <= area_threshold:
                candidates[box.cls].append(Candidate(image=image, label=label, boxes=boxes, target_index=index))
    return candidates


def add_crop_images(
    candidates_by_class: dict[int, list[Candidate]],
    output: Path,
    max_crops_per_class: int,
    crop_scale: float,
    min_crop_side: int,
    jitter: float,
    min_visibility: float,
    quality: int,
    rng: random.Random,
) -> Counter[int]:
    created: Counter[int] = Counter()
    serial = 0
    for cls in sorted(candidates_by_class):
        candidates = candidates_by_class[cls][:]
        rng.shuffle(candidates)
        for candidate in candidates[:max_crops_per_class]:
            with Image.open(candidate.image) as raw_image:
                image = ImageOps.exif_transpose(raw_image).convert("RGB")
            width, height = image.size
            target = candidate.boxes[candidate.target_index]
            window = make_crop_window(target, width, height, crop_scale, min_crop_side, jitter, rng)
            labels = crop_labels(candidate.boxes, width, height, window, min_visibility)
            if not any(box.cls == cls for box in labels):
                continue
            serial += 1
            stem = f"{candidate.image.stem}__crop_c{cls}_{serial:05d}"
            out_image = output / "images" / "train" / f"{stem}.jpg"
            out_label = output / "labels" / "train" / f"{stem}.txt"
            image.crop(window).save(out_image, format="JPEG", quality=quality)
            save_boxes(labels, out_label)
            created[cls] += 1
    return created


def write_data_yaml(output: Path, raw: dict) -> None:
    names = raw.get("names") or {}
    out = {
        "path": output.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    yaml.safe_dump(out, output.joinpath("data.yaml").open("w", encoding="utf-8"), allow_unicode=True, sort_keys=False)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    raw, root = load_data_yaml(data_path)
    output = Path(args.output).resolve()
    crop_classes = {int(item.strip()) for item in args.crop_classes.split(",") if item.strip()}
    rng = random.Random(args.seed)

    train_images = resolve_split(root, raw.get("train", "images/train"))
    val_images = resolve_split(root, raw.get("val", "images/val"))
    prepare_output(output, root, args.overwrite)
    copy_split(train_images, root, output, "train", args.copy_originals)
    copy_split(val_images, root, output, "val", args.copy_originals)
    candidates_by_class = collect_candidates(train_images, root, crop_classes, args.area_threshold)
    created = add_crop_images(
        candidates_by_class,
        output,
        args.max_crops_per_class,
        args.crop_scale,
        args.min_crop_side,
        args.jitter,
        args.min_visibility,
        args.quality,
        rng,
    )
    write_data_yaml(output, raw)

    names = {int(k): str(v) for k, v in (raw.get("names") or {}).items()}
    print(f"source_train_images={len(train_images)}")
    print(f"source_val_images={len(val_images)}")
    print(f"output={output}")
    for cls in sorted(crop_classes):
        print(
            f"class={cls} name={names.get(cls, cls)} candidates={len(candidates_by_class.get(cls, []))} "
            f"created={created[cls]}"
        )
    print(f"crop_images_total={sum(created.values())}")
    print(f"train_images_total={len(train_images) + sum(created.values())}")
    print(f"data_yaml={output / 'data.yaml'}")


if __name__ == "__main__":
    main()

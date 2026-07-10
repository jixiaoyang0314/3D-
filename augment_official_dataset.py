from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance, ImageFile, ImageOps


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OFFICIAL_OBJECTS = {
    "CA001": "刷子",
    "CA002": "耳机",
    "CA003": "水杯",
    "CA004": "衣架",
    "CB001": "巧克力",
    "CB002": "瓜子",
    "CB003": "火腿肠",
    "CB004": "薯片",
    "CC001": "罐装饮料",
    "CC002": "瓶装饮料",
    "CC003": "盒装牛奶",
    "CC004": "瓶装饮用水",
    "CD001": "桃子",
    "CD002": "苹果",
    "CD003": "香蕉",
    "CD004": "梨",
}


@dataclass(frozen=True)
class DatasetItem:
    image: Path
    label: Path
    classes: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a re-encoded, light-augmented, class-balanced YOLO dataset "
            "for the 2026 3D recognition contest."
        )
    )
    parser.add_argument("--data", required=True, help="Source YOLO data.yaml.")
    parser.add_argument(
        "--output",
        help="Output dataset directory. Defaults to '<source>_balanced_aug'.",
    )
    parser.add_argument("--yellow", type=float, default=0.10, help="Yellow light train-image ratio.")
    parser.add_argument("--white", type=float, default=0.10, help="White light train-image ratio.")
    parser.add_argument("--dark", type=float, default=0.05, help="Dark light train-image ratio.")
    parser.add_argument(
        "--balance-target",
        default="max",
        help="Class-balance target: none, median, p75, max, or an integer box count.",
    )
    parser.add_argument(
        "--copy-originals",
        choices=["reencode", "copy", "hardlink"],
        default="reencode",
        help="How to place original images into the output dataset.",
    )
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality for re-encoded images.")
    parser.add_argument(
        "--max-side",
        type=int,
        default=1920,
        help="Resize re-encoded images so the longest side is at most this value. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-balance-images",
        type=int,
        default=6000,
        help="Safety cap for extra balancing images.",
    )
    parser.add_argument(
        "--balance-pick",
        choices=["least-used", "random"],
        default="least-used",
        help="How to choose source images while balancing classes.",
    )
    parser.add_argument(
        "--max-source-repeats",
        type=int,
        default=0,
        help=(
            "Maximum number of total train variants allowed per source image while balancing. "
            "Use 0 to disable the cap."
        ),
    )
    return parser.parse_args()


def load_data_yaml(path: Path) -> tuple[dict, Path]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a mapping-style YOLO data yaml")

    root = Path(raw.get("path") or path.parent)
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return raw, root


def split_dir(root: Path, rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else root / path


def read_classes(label_path: Path) -> tuple[int, ...]:
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        classes.append(int(line.split()[0]))
    return tuple(classes)


def collect_items(root: Path, rel_images: str | Path, rel_labels: str | Path) -> list[DatasetItem]:
    image_dir = split_dir(root, rel_images)
    label_dir = split_dir(root, rel_labels)
    items: list[DatasetItem] = []
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        classes = read_classes(label_path)
        if not classes:
            raise ValueError(f"Empty label file is not expected in this training dataset: {label_path}")
        items.append(DatasetItem(image=image_path, label=label_path, classes=classes))
    return items


def count_classes(items: list[DatasetItem]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for item in items:
        counts.update(item.classes)
    return counts


def base_stem(path: Path) -> str:
    return path.stem.split("__aug_")[0]


def count_base_repeats(items: list[DatasetItem]) -> Counter[str]:
    return Counter(base_stem(item.image) for item in items)


def ensure_empty_output(output: Path, source_root: Path, overwrite: bool) -> None:
    output = output.resolve()
    source_root = source_root.resolve()
    if output == source_root:
        raise ValueError("Refusing to overwrite the source dataset directory.")
    if output.parent == output:
        raise ValueError("Refusing to use a filesystem root as the output dataset directory.")
    if len(output.parts) < 3:
        raise ValueError(f"Refusing to remove suspiciously broad output path: {output}")

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def safe_open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def resize_for_training(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    current_max = max(width, height)
    if current_max <= max_side:
        return image
    scale = max_side / current_max
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def save_jpeg(image: Image.Image, path: Path, quality: int, max_side: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = resize_for_training(image, max_side)
    image.save(path, format="JPEG", quality=quality)


def place_original_image(source: Path, dest_stem: Path, mode: str, quality: int, max_side: int) -> Path:
    if mode == "reencode":
        dest = dest_stem.with_suffix(".jpg")
        save_jpeg(safe_open_rgb(source), dest, quality, max_side)
        return dest

    dest = dest_stem.with_suffix(source.suffix.lower())
    if mode == "hardlink":
        try:
            dest.hardlink_to(source)
            return dest
        except OSError:
            shutil.copy2(source, dest)
            return dest

    shutil.copy2(source, dest)
    return dest


def copy_label(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def write_data_yaml(output: Path, names: dict[int, str]) -> None:
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx in sorted(names):
        official_id = str(names[idx])
        object_name = OFFICIAL_OBJECTS.get(official_id, "")
        comment = f"  # {object_name}" if object_name else ""
        lines.append(f"  {idx}: {official_id}{comment}")
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def multiply_channels(image: Image.Image, factors: tuple[float, float, float]) -> Image.Image:
    r, g, b = image.split()
    channels = []
    for channel, factor in zip((r, g, b), factors):
        channels.append(channel.point(lambda value, f=factor: max(0, min(255, int(value * f)))))
    return Image.merge("RGB", channels)


def add_light_variant(image: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    if kind == "yellow":
        image = multiply_channels(
            image,
            (
                rng.uniform(1.08, 1.18),
                rng.uniform(1.02, 1.08),
                rng.uniform(0.72, 0.88),
            ),
        )
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(1.04, 1.16))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.95, 1.08))
        image = ImageEnhance.Color(image).enhance(rng.uniform(1.02, 1.14))
        return image

    if kind == "white":
        image = multiply_channels(
            image,
            (
                rng.uniform(1.03, 1.12),
                rng.uniform(1.03, 1.12),
                rng.uniform(1.05, 1.16),
            ),
        )
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(1.10, 1.24))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(1.02, 1.12))
        return image

    if kind == "dark":
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.45, 0.65))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.00))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.82, 0.96))
        return image

    if kind == "balance":
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.78, 1.22))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.14))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.86, 1.18))
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.85, 1.25))
        return image

    raise ValueError(f"Unsupported light variant: {kind}")


def make_augmented_item(
    item: DatasetItem,
    output: Path,
    kind: str,
    serial: int,
    rng: random.Random,
    quality: int,
    max_side: int,
) -> DatasetItem:
    suffix = f"__aug_{kind}_{serial:05d}"
    dest_image = output / "images" / "train" / f"{item.image.stem}{suffix}.jpg"
    dest_label = output / "labels" / "train" / f"{item.image.stem}{suffix}.txt"
    image = add_light_variant(safe_open_rgb(item.image), kind, rng)
    save_jpeg(image, dest_image, quality, max_side)
    copy_label(item.label, dest_label)
    return DatasetItem(image=dest_image, label=dest_label, classes=item.classes)


def compute_balance_target(counts: Counter[int], target: str) -> int | None:
    values = sorted(counts.values())
    if target == "none":
        return None
    if target == "median":
        return values[len(values) // 2]
    if target == "p75":
        return values[int((len(values) - 1) * 0.75)]
    if target == "max":
        return max(values)
    try:
        parsed = int(target)
    except ValueError as exc:
        raise ValueError("--balance-target must be none, median, p75, max, or an integer") from exc
    if parsed <= 0:
        raise ValueError("--balance-target integer must be positive")
    return parsed


def add_light_augmentations(
    train_items: list[DatasetItem],
    output: Path,
    ratios: dict[str, float],
    rng: random.Random,
    quality: int,
    max_side: int,
) -> tuple[list[DatasetItem], dict[str, int]]:
    shuffled = train_items[:]
    rng.shuffle(shuffled)
    cursor = 0
    created: list[DatasetItem] = []
    counts: dict[str, int] = {}
    for kind, ratio in ratios.items():
        if ratio < 0:
            raise ValueError(f"{kind} ratio must be >= 0")
        requested = int(round(len(train_items) * ratio))
        selected = shuffled[cursor : cursor + requested]
        cursor += requested
        counts[kind] = len(selected)
        for local_idx, item in enumerate(selected, 1):
            serial = len(created) + local_idx
            created.append(make_augmented_item(item, output, kind, serial, rng, quality, max_side))
    return created, counts


def add_balance_augmentations(
    train_items: list[DatasetItem],
    existing_items: list[DatasetItem],
    output: Path,
    names: dict[int, str],
    target_arg: str,
    rng: random.Random,
    quality: int,
    max_side: int,
    max_images: int,
    balance_pick: str,
    max_source_repeats: int,
) -> tuple[list[DatasetItem], int | None]:
    counts = count_classes(existing_items)
    target = compute_balance_target(counts, target_arg)
    if target is None:
        return [], None

    by_class: dict[int, list[DatasetItem]] = {idx: [] for idx in names}
    for item in train_items:
        for cls in set(item.classes):
            by_class.setdefault(cls, []).append(item)

    created: list[DatasetItem] = []
    source_usage = count_base_repeats(existing_items)
    while created.__len__() < max_images:
        deficits = {idx: target - counts.get(idx, 0) for idx in names}
        cls, deficit = max(deficits.items(), key=lambda item: item[1])
        if deficit <= 0:
            break
        candidates = by_class.get(cls) or []
        if not candidates:
            raise ValueError(f"No source images are available for class {cls} ({names[cls]})")

        if max_source_repeats > 0:
            capped_candidates = [
                item for item in candidates if source_usage[base_stem(item.image)] < max_source_repeats
            ]
            if capped_candidates:
                candidates = capped_candidates
            else:
                raise RuntimeError(
                    f"Class {cls} ({names[cls]}) still needs {deficit} boxes, but all source images "
                    f"hit --max-source-repeats={max_source_repeats}."
                )

        if balance_pick == "least-used":
            min_usage = min(source_usage[base_stem(item.image)] for item in candidates)
            least_used = [item for item in candidates if source_usage[base_stem(item.image)] == min_usage]
            item = rng.choice(least_used)
        else:
            item = rng.choice(candidates)

        created_item = make_augmented_item(
            item,
            output,
            "balance",
            len(created) + 1,
            rng,
            quality,
            max_side,
        )
        created.append(created_item)
        counts.update(created_item.classes)
        source_usage[base_stem(item.image)] += 1

    if max(counts.get(idx, 0) < target for idx in names):
        raise RuntimeError(
            f"Reached --max-balance-images={max_images} before all classes hit target {target}."
        )
    return created, target


def copy_original_split(
    items: list[DatasetItem],
    output: Path,
    split: str,
    copy_mode: str,
    quality: int,
    max_side: int,
) -> list[DatasetItem]:
    copied: list[DatasetItem] = []
    for item in items:
        dest_stem = output / "images" / split / item.image.stem
        dest_image = place_original_image(item.image, dest_stem, copy_mode, quality, max_side)
        dest_label = output / "labels" / split / f"{item.image.stem}.txt"
        copy_label(item.label, dest_label)
        copied.append(DatasetItem(image=dest_image, label=dest_label, classes=item.classes))
    return copied


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    raw, source_root = load_data_yaml(data_path)
    output = Path(args.output).resolve() if args.output else source_root.with_name(f"{source_root.name}_balanced_aug")
    names = {int(idx): str(name) for idx, name in (raw.get("names") or {}).items()}

    train_rel = raw.get("train", "images/train")
    val_rel = raw.get("val", "images/val")
    train_items = collect_items(source_root, train_rel, Path(train_rel).parent.parent / "labels" / Path(train_rel).name)
    val_items = collect_items(source_root, val_rel, Path(val_rel).parent.parent / "labels" / Path(val_rel).name)

    rng = random.Random(args.seed)
    ensure_empty_output(output, source_root, args.overwrite)

    copied_train = copy_original_split(
        train_items,
        output,
        "train",
        args.copy_originals,
        args.quality,
        args.max_side,
    )
    copied_val = copy_original_split(
        val_items,
        output,
        "val",
        args.copy_originals,
        args.quality,
        args.max_side,
    )
    light_items, light_counts = add_light_augmentations(
        train_items,
        output,
        {"yellow": args.yellow, "white": args.white, "dark": args.dark},
        rng,
        args.quality,
        args.max_side,
    )
    balance_items, balance_target = add_balance_augmentations(
        train_items,
        copied_train + light_items,
        output,
        names,
        args.balance_target,
        rng,
        args.quality,
        args.max_side,
        args.max_balance_images,
        args.balance_pick,
        args.max_source_repeats,
    )

    final_train = copied_train + light_items + balance_items
    write_data_yaml(output, names)

    summary = {
        "source": str(source_root),
        "output": str(output),
        "seed": args.seed,
        "copy_originals": args.copy_originals,
        "quality": args.quality,
        "max_side": args.max_side,
        "balance_pick": args.balance_pick,
        "max_source_repeats": args.max_source_repeats,
        "train_images_original": len(train_items),
        "val_images_original": len(val_items),
        "light_augmented_images": light_counts,
        "balance_target": balance_target,
        "balance_augmented_images": len(balance_items),
        "train_images_final": len(final_train),
        "val_images_final": len(copied_val),
        "class_counts_before_train": dict(sorted(count_classes(train_items).items())),
        "class_counts_after_train": dict(sorted(count_classes(final_train).items())),
        "max_source_repeats_after_train": max(count_base_repeats(final_train).values()),
        "names": names,
    }
    (output / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

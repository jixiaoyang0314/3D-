from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(slots=True)
class Sample:
    image_path: Path
    label_lines: list[str]
    source_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the local annotated dataset as a YOLO dataset.")
    parser.add_argument("--source", default="D:/机器人识别大赛/data/data", help="Raw data directory.")
    parser.add_argument("--output", default="D:/机器人识别大赛/yolo_dataset", help="YOLO dataset output directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--link", choices=["hardlink", "copy", "symlink"], default="hardlink")
    return parser.parse_args()


def source_group(name: str) -> str:
    if re.fullmatch(r"part\d+", name, flags=re.IGNORECASE):
        return "part"
    return name


def find_label_zip(root: Path, source_name: str) -> Path | None:
    candidates = [
        root / f"{source_name}.zip",
        root / f"{source_name}_with_label.zip",
        root / f"{source_name}_label.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def image_by_stem(image_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images[path.stem] = path
    return images


def read_zip_labels(label_zip: Path) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    with ZipFile(label_zip) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".txt"):
                continue
            stem = Path(name).stem
            text = archive.read(name).decode("utf-8", errors="replace")
            labels[stem] = [line.strip() for line in text.splitlines() if line.strip()]
    return labels


def discover_samples(root: Path) -> tuple[list[Sample], list[str], list[str]]:
    warnings: list[str] = []
    label_cache: dict[str, dict[str, list[str]]] = {}
    local_to_global: dict[tuple[str, str], int] = {}
    class_names: list[str] = []
    samples: list[Sample] = []

    for image_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label_zip = find_label_zip(root, image_dir.name)
        if label_zip is None:
            warnings.append(f"skip {image_dir.name}: no matching label zip")
            continue

        labels = read_zip_labels(label_zip)
        label_cache[image_dir.name] = labels
        images = image_by_stem(image_dir)
        if not images:
            warnings.append(f"skip {image_dir.name}: no images")
            continue

        group = source_group(image_dir.name)
        local_ids = sorted(
            {
                line.split()[0]
                for lines in labels.values()
                for line in lines
                if line.split()
            },
            key=lambda item: int(item) if item.isdigit() else item,
        )
        single_class = len(local_ids) == 1

        for stem, image_path in images.items():
            source_lines = labels.get(stem)
            if not source_lines:
                continue
            rewritten: list[str] = []
            for line in source_lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                local_id = parts[0]
                key = (group, local_id)
                if key not in local_to_global:
                    class_name = group if single_class else f"{group}_{local_id}"
                    local_to_global[key] = len(class_names)
                    class_names.append(class_name)
                rewritten.append(" ".join([str(local_to_global[key]), *parts[1:]]))
            if rewritten:
                samples.append(Sample(image_path=image_path, label_lines=rewritten, source_name=image_dir.name))

    return samples, class_names, warnings


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def write_split(samples: list[Sample], output: Path, split: str, mode: str) -> None:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples):
        out_stem = f"{sample.source_name}_{sample.image_path.stem}_{index:06d}"
        image_out = image_dir / f"{out_stem}{sample.image_path.suffix.lower()}"
        label_out = label_dir / f"{out_stem}.txt"
        link_or_copy(sample.image_path, image_out, mode)
        label_out.write_text("\n".join(sample.label_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory already exists and is not empty: {output}")

    samples, class_names, warnings = discover_samples(source)
    if not samples:
        raise RuntimeError("No labeled samples were discovered.")

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    val_count = max(1, int(round(len(samples) * args.val_ratio)))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    write_split(train_samples, output, "train", args.link)
    write_split(val_samples, output, "val", args.link)

    data_yaml = {
        # Keep the dataset yaml portable and avoid Windows console encoding
        # issues with non-ASCII absolute paths.
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"samples={len(samples)} train={len(train_samples)} val={len(val_samples)} classes={len(class_names)}")
    print(f"data_yaml={output / 'data.yaml'}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()

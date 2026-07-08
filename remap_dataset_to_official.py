from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml


OLD_NAME_TO_OFFICIAL_ID = {
    "banana": "CD003",
    "bb_0": "CB004",
    "bb_1": "CC004",
    "bb_2": "CC003",
    "bb_3": "CB003",
    "cup": "CA003",
    "new_0": "CB002",
    "new_1": "CB003",
    "new_2": "CC003",
    "part_0": "CA002",
    "part_1": "CC002",
    "part_2": "CC004",
    "part_3": "CB004",
    "part_4": "CB002",
    "part_5": "CB003",
    "part_6": "CC001",
    "part_7": "CC003",
    "pinzi": "CC004",
    "shupian+huotuichang_0": "CB003",
    "shupian+huotuichang_1": "CB004",
    "yijia": "CA004",
}

OFFICIAL_IDS = [
    "CA002",
    "CA003",
    "CA004",
    "CB002",
    "CB003",
    "CB004",
    "CC001",
    "CC002",
    "CC003",
    "CC004",
    "CD003",
]

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge generated YOLO classes into official competition IDs.")
    parser.add_argument("--source", default="../yolo_dataset", help="Source YOLO dataset directory.")
    parser.add_argument("--output", default="../yolo_dataset_official", help="Output remapped YOLO dataset directory.")
    parser.add_argument("--link", choices=["hardlink", "copy", "symlink"], default="hardlink")
    return parser.parse_args()


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


def load_source_names(source: Path) -> dict[int, str]:
    data = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def remap_label_file(
    label_path: Path,
    old_index_to_new_index: dict[int, int],
) -> tuple[list[str], int, int]:
    output_lines: list[str] = []
    skipped_unknown = 0
    skipped_invalid = 0

    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            skipped_invalid += 1
            continue
        try:
            old_index = int(float(parts[0]))
            x, y, w, h = (float(value) for value in parts[1:])
        except ValueError:
            skipped_invalid += 1
            continue
        if w <= 0.0 or h <= 0.0:
            skipped_invalid += 1
            continue
        if old_index not in old_index_to_new_index:
            skipped_unknown += 1
            continue
        output_lines.append(" ".join([str(old_index_to_new_index[old_index]), f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"]))

    return output_lines, skipped_unknown, skipped_invalid


def write_data_yaml(output: Path) -> None:
    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {idx: official_id for idx, official_id in enumerate(OFFICIAL_IDS)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    if not (source / "data.yaml").exists():
        raise FileNotFoundError(source / "data.yaml")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory already exists and is not empty: {output}")

    source_names = load_source_names(source)
    official_to_new_index = {official_id: idx for idx, official_id in enumerate(OFFICIAL_IDS)}
    old_index_to_new_index = {
        old_index: official_to_new_index[official_id]
        for old_index, old_name in source_names.items()
        if (official_id := OLD_NAME_TO_OFFICIAL_ID.get(old_name)) is not None
    }

    total_images = 0
    total_boxes = 0
    skipped_unknown = 0
    skipped_invalid = 0

    for split in ("train", "val"):
        source_image_dir = source / "images" / split
        source_label_dir = source / "labels" / split
        output_image_dir = output / "images" / split
        output_label_dir = output / "labels" / split
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for label_path in sorted(source_label_dir.glob("*.txt")):
            image_path = find_image(source_image_dir, label_path.stem)
            if image_path is None:
                continue
            label_lines, unknown_count, invalid_count = remap_label_file(label_path, old_index_to_new_index)
            skipped_unknown += unknown_count
            skipped_invalid += invalid_count
            if not label_lines:
                continue

            image_out = output_image_dir / image_path.name
            label_out = output_label_dir / label_path.name
            link_or_copy(image_path, image_out, args.link)
            label_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            total_images += 1
            total_boxes += len(label_lines)

    write_data_yaml(output)
    print(f"output={output}")
    print(f"images={total_images} boxes={total_boxes} classes={len(OFFICIAL_IDS)}")
    print(f"skipped_unknown={skipped_unknown} skipped_invalid={skipped_invalid}")


if __name__ == "__main__":
    main()

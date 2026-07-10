from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_WEAK_CLASSES = (0, 3, 4, 6, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an oversampled YOLO train list for weak classes.")
    parser.add_argument("--data", required=True, help="Source YOLO data.yaml.")
    parser.add_argument("--output", required=True, help="Output data yaml.")
    parser.add_argument("--train-list", required=True, help="Output train image list.")
    parser.add_argument("--weak-classes", default=",".join(map(str, DEFAULT_WEAK_CLASSES)))
    parser.add_argument("--weak-copies", type=int, default=3, help="Total copies for weak-class images.")
    return parser.parse_args()


def read_classes(label_path: Path) -> set[int]:
    classes: set[int] = set()
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        classes.add(int(line.split()[0]))
    return classes


def image_to_label(image_path: Path, root: Path) -> Path:
    rel = image_path.relative_to(root / "images" / "train")
    return root / "labels" / "train" / rel.with_suffix(".txt")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    raw = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    root = Path(raw.get("path") or data_path.parent)
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()

    weak_classes = {int(item.strip()) for item in args.weak_classes.split(",") if item.strip()}
    image_dir = root / "images" / "train"
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    output_lines: list[str] = []
    weak_image_count = 0
    for image_path in image_paths:
        label_classes = read_classes(image_to_label(image_path, root))
        copies = args.weak_copies if label_classes & weak_classes else 1
        if copies > 1:
            weak_image_count += 1
        output_lines.extend(image_path.as_posix() for _ in range(copies))

    train_list_path = Path(args.train_list)
    train_list_path.parent.mkdir(parents=True, exist_ok=True)
    train_list_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    out_yaml = dict(raw)
    out_yaml["path"] = root.as_posix()
    out_yaml["train"] = train_list_path.resolve().as_posix()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(out_yaml, output_path.open("w", encoding="utf-8"), allow_unicode=True, sort_keys=False)

    print(f"source_images={len(image_paths)}")
    print(f"weak_images={weak_image_count}")
    print(f"train_entries={len(output_lines)}")
    print(f"train_list={train_list_path}")
    print(f"data_yaml={output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    class_id: int
    official_id: str
    name_cn: str
    confidence: float
    xyxy: list[float]


def check_ret(ret: int, message: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{message} failed, ret={ret}")


def acl_ret(value) -> int:
    if isinstance(value, tuple):
        return int(value[-1])
    return int(value)


def load_labels(path: Path) -> tuple[list[str], list[str]]:
    official_ids: list[str] = []
    names_cn: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 2:
            raise ValueError(f"Bad label line: {line}")
        official_ids.append(parts[1])
        names_cn.append(parts[2] if len(parts) >= 3 else parts[1])
    return official_ids, names_cn


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    images = [p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    new_w, new_h = int(round(width * gain)), int(round(height * gain))
    pad_w = (size - new_w) / 2.0
    pad_h = (size - new_h) / 2.0

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, gain, (pad_w, pad_h)


def preprocess(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    padded, gain, pad = letterbox(image, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), gain, pad


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def nms_class_aware(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    max_det: int,
) -> list[int]:
    keep: list[int] = []
    for class_id in np.unique(classes):
        idxs = np.where(classes == class_id)[0]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        while idxs.size:
            current = int(idxs[0])
            keep.append(current)
            if len(keep) >= max_det or idxs.size == 1:
                break
            ious = box_iou(boxes[current], boxes[idxs[1:]])
            idxs = idxs[1:][ious <= iou_threshold]
    keep.sort(key=lambda i: float(scores[i]), reverse=True)
    return keep[:max_det]


def decode_yolo_output(
    raw_output: np.ndarray,
    original_shape: tuple[int, int],
    gain: float,
    pad: tuple[float, float],
    official_ids: list[str],
    names_cn: list[str],
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> list[Detection]:
    num_classes = len(official_ids)
    channels = 4 + num_classes
    pred = raw_output.reshape(1, channels, -1)[0]

    boxes_xywh = pred[:4].T
    class_scores = pred[4:].T
    classes = class_scores.argmax(axis=1).astype(np.int32)
    scores = class_scores[np.arange(class_scores.shape[0]), classes]

    mask = scores >= conf_threshold
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    classes = classes[mask]

    boxes = np.empty_like(boxes_xywh, dtype=np.float32)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    pad_w, pad_h = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain
    height, width = original_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

    keep = nms_class_aware(boxes, scores, classes, iou_threshold, max_det)
    detections: list[Detection] = []
    for idx in keep:
        class_id = int(classes[idx])
        detections.append(
            Detection(
                class_id=class_id,
                official_id=official_ids[class_id],
                name_cn=names_cn[class_id],
                confidence=float(scores[idx]),
                xyxy=[float(v) for v in boxes[idx]],
            )
        )
    return detections


class AclOmModel:
    def __init__(self, model_path: Path, device_id: int = 0) -> None:
        try:
            import acl  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Cannot import acl. Source CANN set_env.sh and install pyACL first.") from exc

        self.acl = acl
        self.device_id = device_id
        self.model_path = str(model_path)
        self.model_id: int | None = None
        self.model_desc = None
        self.context = None
        self.input_buffer = None
        self.output_buffers: list[int] = []
        self.input_dataset = None
        self.output_dataset = None
        self.input_size = 0
        self.output_sizes: list[int] = []

        check_ret(self.acl.init(), "acl.init")
        check_ret(self.acl.rt.set_device(device_id), "acl.rt.set_device")
        self.context, ret = self.acl.rt.create_context(device_id)
        check_ret(ret, "acl.rt.create_context")
        self.model_id, ret = self.acl.mdl.load_from_file(self.model_path)
        check_ret(ret, "acl.mdl.load_from_file")

        self.model_desc = self.acl.mdl.create_desc()
        check_ret(self.acl.mdl.get_desc(self.model_desc, self.model_id), "acl.mdl.get_desc")
        self._allocate_io()

    def _allocate_io(self) -> None:
        self.input_dataset = self.acl.mdl.create_dataset()
        self.output_dataset = self.acl.mdl.create_dataset()

        self.input_size = int(self.acl.mdl.get_input_size_by_index(self.model_desc, 0))
        self.input_buffer, ret = self.acl.rt.malloc(self.input_size, ACL_MEM_MALLOC_HUGE_FIRST)
        check_ret(ret, "acl.rt.malloc input")
        input_data = self.acl.create_data_buffer(self.input_buffer, self.input_size)
        check_ret(acl_ret(self.acl.mdl.add_dataset_buffer(self.input_dataset, input_data)), "acl.mdl.add_dataset_buffer input")

        output_count = int(self.acl.mdl.get_num_outputs(self.model_desc))
        for idx in range(output_count):
            size = int(self.acl.mdl.get_output_size_by_index(self.model_desc, idx))
            buffer, ret = self.acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret(ret, f"acl.rt.malloc output {idx}")
            data = self.acl.create_data_buffer(buffer, size)
            check_ret(acl_ret(self.acl.mdl.add_dataset_buffer(self.output_dataset, data)), f"acl.mdl.add_dataset_buffer output {idx}")
            self.output_buffers.append(buffer)
            self.output_sizes.append(size)

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        input_bytes = tensor.tobytes()
        if len(input_bytes) != self.input_size:
            raise ValueError(f"Input byte size {len(input_bytes)} does not match OM input size {self.input_size}.")
        input_ptr = self.acl.util.bytes_to_ptr(input_bytes)
        check_ret(
            self.acl.rt.memcpy(
                self.input_buffer,
                self.input_size,
                input_ptr,
                self.input_size,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "acl.rt.memcpy input",
        )
        check_ret(self.acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset), "acl.mdl.execute")

        outputs: list[np.ndarray] = []
        for buffer, size in zip(self.output_buffers, self.output_sizes):
            host_ptr, ret = self.acl.rt.malloc_host(size)
            check_ret(ret, "acl.rt.malloc_host output")
            try:
                check_ret(
                    self.acl.rt.memcpy(host_ptr, size, buffer, size, ACL_MEMCPY_DEVICE_TO_HOST),
                    "acl.rt.memcpy output",
                )
                output_bytes = self.acl.util.ptr_to_bytes(host_ptr, size)
            finally:
                self.acl.rt.free_host(host_ptr)
            outputs.append(array_from_output_bytes(output_bytes, size))
        return outputs

    def close(self) -> None:
        if self.output_dataset is not None:
            destroy_dataset(self.acl, self.output_dataset)
            self.output_dataset = None
        if self.input_dataset is not None:
            destroy_dataset(self.acl, self.input_dataset)
            self.input_dataset = None
        for buffer in self.output_buffers:
            self.acl.rt.free(buffer)
        self.output_buffers.clear()
        if self.input_buffer is not None:
            self.acl.rt.free(self.input_buffer)
            self.input_buffer = None
        if self.model_desc is not None:
            self.acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        if self.model_id is not None:
            self.acl.mdl.unload(self.model_id)
            self.model_id = None
        if self.context is not None:
            self.acl.rt.destroy_context(self.context)
            self.context = None
        self.acl.rt.reset_device(self.device_id)
        self.acl.finalize()


def array_from_output_bytes(output_bytes: bytes, byte_size: int) -> np.ndarray:
    expected_values = (4 + 11) * 8400
    if byte_size % np.dtype(np.float32).itemsize == 0:
        values = np.frombuffer(output_bytes, dtype=np.float32)
        if values.size >= expected_values:
            return values[:expected_values].copy()
    if byte_size % np.dtype(np.float16).itemsize == 0:
        values = np.frombuffer(output_bytes, dtype=np.float16)
        if values.size >= expected_values:
            return values[:expected_values].astype(np.float32)
    raise ValueError(f"Unsupported output byte size: {byte_size}")


def destroy_dataset(acl, dataset) -> None:
    count = acl.mdl.get_dataset_num_buffers(dataset)
    for idx in range(count):
        data_buffer = acl.mdl.get_dataset_buffer(dataset, idx)
        if data_buffer:
            acl.destroy_data_buffer(data_buffer)
    acl.mdl.destroy_dataset(dataset)


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.xyxy]
        color = color_for_class(det.class_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{det.official_id} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(canvas, (x1, y_text), (x1 + tw + 6, y_text + th + baseline + 6), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y_text + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 132, 255),
        (80, 180, 80),
        (245, 150, 60),
        (180, 90, 220),
        (60, 180, 190),
        (220, 80, 80),
        (130, 160, 50),
        (70, 120, 200),
        (200, 120, 50),
        (120, 80, 200),
        (50, 170, 120),
    ]
    return palette[class_id % len(palette)]


def write_txt(path: Path, detections: list[Detection]) -> None:
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        lines.append(f"{det.official_id} {det.confidence:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO OM inference through Ascend pyACL.")
    parser.add_argument("--model", default="weights/best.om", help="Path to .om model.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--labels", default="labels.txt", help="Label mapping file.")
    parser.add_argument("--out-dir", default="runs_om", help="Output directory.")
    parser.add_argument("--device", type=int, default=0, help="Ascend device id.")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image.")
    parser.add_argument("--save-vis", action="store_true", help="Save visualized images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    model_path = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    source = (root / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source)
    labels_path = (root / args.labels).resolve() if not Path(args.labels).is_absolute() else Path(args.labels)
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    label_dir = out_dir / "labels"
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    official_ids, names_cn = load_labels(labels_path)
    images = iter_images(source)
    if not images:
        raise FileNotFoundError(f"No images found in {source}")

    model = AclOmModel(model_path, args.device)
    all_results = []
    try:
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"skip unreadable image: {image_path}", file=sys.stderr)
                continue
            tensor, gain, pad = preprocess(image, args.imgsz)
            start = time.perf_counter()
            outputs = model.infer(tensor)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            detections = decode_yolo_output(
                outputs[0],
                image.shape[:2],
                gain,
                pad,
                official_ids,
                names_cn,
                args.conf,
                args.iou,
                args.max_det,
            )
            write_txt(label_dir / f"{image_path.stem}.txt", detections)
            if args.save_vis:
                cv2.imwrite(str(vis_dir / image_path.name), draw_detections(image, detections))
            item = {
                "image": str(image_path),
                "elapsed_ms": elapsed_ms,
                "detections": [asdict(det) for det in detections],
            }
            all_results.append(item)
            print(f"{image_path.name}: {len(detections)} detections, {elapsed_ms:.1f} ms")
    finally:
        model.close()

    (out_dir / "predictions.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

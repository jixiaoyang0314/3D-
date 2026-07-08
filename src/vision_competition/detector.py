from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Detection


OFFICIAL_CLASS_ALIASES = {
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


class YOLODetector:
    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        imgsz: int = 1280,
        conf: float = 0.12,
        iou: float = 0.60,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics first: pip install ultralytics") from exc

        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = self.model.names

    def predict(self, image: np.ndarray, frame_index: int) -> list[Detection]:
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        masks = None
        if getattr(result, "masks", None) is not None and result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        detections: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i, (box, score, cls_id) in enumerate(zip(xyxy, confs, cls_ids, strict=False)):
            raw_class_name = str(self.names.get(int(cls_id), cls_id))
            class_name = OFFICIAL_CLASS_ALIASES.get(raw_class_name, raw_class_name)
            mask = masks[i] if masks is not None and i < len(masks) else None
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(score),
                    xyxy=tuple(float(x) for x in box),
                    mask=mask,
                    frame_index=frame_index,
                )
            )
        return detections

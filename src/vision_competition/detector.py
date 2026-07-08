from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Detection


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
            class_name = str(self.names.get(int(cls_id), cls_id))
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


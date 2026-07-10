from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .geometry import box_iou
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
        weights: str | Path | list[str | Path],
        device: str = "cpu",
        imgsz: int = 1280,
        conf: float = 0.12,
        iou: float = 0.60,
        multiscale_imgsz: list[int] | None = None,
        wbf_iou: float = 0.55,
        refine_enabled: bool = False,
        refine_classes: list[str] | None = None,
        refine_area_ratio: float = 0.04,
        refine_max_detections: int = 8,
        refine_crop_scale: float = 3.0,
        refine_min_crop_side: int = 320,
        refine_imgsz: int = 704,
        refine_conf: float = 0.08,
        refine_match_iou: float = 0.12,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics first: pip install ultralytics") from exc

        if isinstance(weights, list):
            weight_list = [str(weight) for weight in weights]
        else:
            weight_text = str(weights)
            weight_list = [item.strip() for item in weight_text.split(",") if item.strip()]
        self.models = [YOLO(weight) for weight in weight_list]
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = self.models[0].names
        scales = multiscale_imgsz or [imgsz]
        self.multiscale_imgsz = list(dict.fromkeys(int(scale) for scale in scales if int(scale) > 0))
        if not self.multiscale_imgsz:
            self.multiscale_imgsz = [imgsz]
        self.wbf_iou = wbf_iou
        self.refine_enabled = refine_enabled
        self.refine_classes = set(refine_classes or [])
        self.refine_area_ratio = refine_area_ratio
        self.refine_max_detections = refine_max_detections
        self.refine_crop_scale = refine_crop_scale
        self.refine_min_crop_side = refine_min_crop_side
        self.refine_imgsz = refine_imgsz
        self.refine_conf = refine_conf
        self.refine_match_iou = refine_match_iou

    def predict(self, image: np.ndarray, frame_index: int) -> list[Detection]:
        detections: list[Detection] = []
        for scale in self.multiscale_imgsz:
            detections.extend(self._predict_once(image, frame_index, imgsz=scale))
        detections = self._weighted_boxes_fusion(detections)
        if self.refine_enabled:
            detections = self._refine_small_objects(image, detections, frame_index)
        return detections

    def _predict_once(self, image: np.ndarray, frame_index: int, imgsz: int) -> list[Detection]:
        detections: list[Detection] = []
        for model_index, model in enumerate(self.models):
            results = model.predict(
                source=image,
                imgsz=imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            if not results:
                continue

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            masks = None
            if getattr(result, "masks", None) is not None and result.masks is not None:
                masks = result.masks.data.cpu().numpy()

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
                        extra={"source": f"model_{model_index}_imgsz_{imgsz}"},
                    )
                )
        return detections

    def _predict_crop(
        self,
        crop: np.ndarray,
        crop_origin: tuple[int, int],
        frame_index: int,
    ) -> list[Detection]:
        results = self.models[0].predict(
            source=crop,
            imgsz=self.refine_imgsz,
            conf=self.refine_conf,
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

        ox, oy = crop_origin
        detections: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, confs, cls_ids, strict=False):
            raw_class_name = str(self.names.get(int(cls_id), cls_id))
            class_name = OFFICIAL_CLASS_ALIASES.get(raw_class_name, raw_class_name)
            x1, y1, x2, y2 = (float(box[0]) + ox, float(box[1]) + oy, float(box[2]) + ox, float(box[3]) + oy)
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(score),
                    xyxy=(x1, y1, x2, y2),
                    frame_index=frame_index,
                    extra={"source": "small_object_refine"},
                )
            )
        return detections

    def _weighted_boxes_fusion(self, detections: list[Detection]) -> list[Detection]:
        if len(detections) <= 1:
            return detections

        groups: list[list[Detection]] = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            best_group: list[Detection] | None = None
            best_iou = 0.0
            for group in groups:
                if group[0].class_name != det.class_name:
                    continue
                iou = box_iou(self._weighted_box(group), det.xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_group = group
            if best_group is not None and best_iou >= self.wbf_iou:
                best_group.append(det)
            else:
                groups.append([det])

        fused: list[Detection] = []
        for group in groups:
            weights = np.asarray([max(det.confidence, 1e-6) for det in group], dtype=np.float32)
            boxes = np.asarray([det.xyxy for det in group], dtype=np.float32)
            xyxy = tuple(float(v) for v in np.average(boxes, axis=0, weights=weights))
            best = max(group, key=lambda item: item.confidence)
            confidence = float(min(1.0, best.confidence + 0.025 * max(0, len(group) - 1)))
            extra = dict(best.extra)
            extra["wbf_sources"] = len(group)
            fused.append(replace(best, confidence=confidence, xyxy=xyxy, extra=extra))
        return fused

    @staticmethod
    def _weighted_box(group: list[Detection]) -> tuple[float, float, float, float]:
        weights = np.asarray([max(det.confidence, 1e-6) for det in group], dtype=np.float32)
        boxes = np.asarray([det.xyxy for det in group], dtype=np.float32)
        return tuple(float(v) for v in np.average(boxes, axis=0, weights=weights))

    def _refine_small_objects(
        self,
        image: np.ndarray,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Detection]:
        if not detections:
            return detections

        height, width = image.shape[:2]
        image_area = max(1.0, float(width * height))
        candidates = [
            (index, det)
            for index, det in enumerate(detections)
            if det.class_name in self.refine_classes or det.area / image_area <= self.refine_area_ratio
        ]
        candidates = sorted(candidates, key=lambda item: item[1].area)[: self.refine_max_detections]
        if not candidates:
            return detections

        refined_by_index: dict[int, Detection] = {}
        for index, det in candidates:
            crop_box = self._make_refine_crop(det.xyxy, width, height)
            x1, y1, x2, y2 = crop_box
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            crop_dets = self._predict_crop(crop, (x1, y1), frame_index)
            match = self._best_refine_match(det, crop_dets)
            if match is None:
                continue
            fused = self._fuse_refined_detection(det, match)
            refined_by_index[index] = fused

        if not refined_by_index:
            return detections
        return [refined_by_index.get(index, det) for index, det in enumerate(detections)]

    def _make_refine_crop(
        self,
        xyxy: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = xyxy
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        side = max(float(self.refine_min_crop_side), max(box_w, box_h) * self.refine_crop_scale)
        crop_w = min(float(width), side)
        crop_h = min(float(height), side)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        left = int(round(max(0.0, min(float(width) - crop_w, cx - crop_w * 0.5))))
        top = int(round(max(0.0, min(float(height) - crop_h, cy - crop_h * 0.5))))
        return left, top, int(round(left + crop_w)), int(round(top + crop_h))

    def _best_refine_match(self, original: Detection, crop_dets: list[Detection]) -> Detection | None:
        best: Detection | None = None
        best_score = 0.0
        ox1, oy1, ox2, oy2 = original.xyxy
        ocx = (ox1 + ox2) * 0.5
        ocy = (oy1 + oy2) * 0.5
        norm = max(np.sqrt(max(original.area, 1.0)), 1.0)
        for det in crop_dets:
            if det.class_name != original.class_name:
                continue
            iou = box_iou(original.xyxy, det.xyxy)
            dx1, dy1, dx2, dy2 = det.xyxy
            dcx = (dx1 + dx2) * 0.5
            dcy = (dy1 + dy2) * 0.5
            center_score = max(0.0, 1.0 - float(np.hypot(dcx - ocx, dcy - ocy)) / (2.5 * norm))
            score = max(iou, center_score * 0.65) * det.confidence
            if iou >= self.refine_match_iou and score > best_score:
                best = det
                best_score = score
        return best

    @staticmethod
    def _fuse_refined_detection(original: Detection, refined: Detection) -> Detection:
        original_weight = max(original.confidence, 1e-6)
        refined_weight = max(refined.confidence, 1e-6) * 0.85
        boxes = np.asarray([original.xyxy, refined.xyxy], dtype=np.float32)
        weights = np.asarray([original_weight, refined_weight], dtype=np.float32)
        xyxy = tuple(float(v) for v in np.average(boxes, axis=0, weights=weights))
        confidence = float(min(1.0, max(original.confidence, refined.confidence) + 0.015))
        extra = dict(original.extra)
        extra["refined"] = True
        extra["refine_confidence"] = refined.confidence
        return replace(original, confidence=confidence, xyxy=xyxy, extra=extra)

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from .config import OCRConfig


@dataclass(slots=True)
class OCRResult:
    object_id: str
    text: str
    confidence: float


class UnknownOCR:
    def __init__(self, config: OCRConfig, enabled: bool = False) -> None:
        self.config = config
        self.enabled = enabled
        self.reader = None
        if enabled:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("Install paddleocr or set ocr_enabled=false") from exc
            self.reader = PaddleOCR(use_angle_cls=True, lang="ch")

    def recognize(self, crop: np.ndarray | None) -> OCRResult | None:
        results = self.recognize_many([] if crop is None else [crop])
        return results

    def recognize_many(self, crops: list[np.ndarray]) -> OCRResult | None:
        if not self.enabled or self.reader is None:
            return None
        votes: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            for variant in self._variants(crop):
                text, confidence = self._read_text(variant)
                if not text:
                    continue
                object_id = self.map_text_to_id(text)
                if object_id is not None:
                    votes[object_id].append((text, confidence))

        best_id = ""
        best_score = 0.0
        best_text = ""
        for object_id, object_votes in votes.items():
            if len(object_votes) < self.config.vote_min_count:
                continue
            score = max(conf for _, conf in object_votes)
            if score > best_score:
                best_id = object_id
                best_score = score
                best_text = " ".join(text for text, _ in object_votes[:3])

        if not best_id or best_score < self.config.min_confidence:
            return None
        return OCRResult(object_id=best_id, text=best_text, confidence=best_score)

    def _read_text(self, crop: np.ndarray) -> tuple[str, float]:
        result = self.reader.ocr(crop, cls=True)
        texts: list[str] = []
        confidences: list[float] = []
        for line_group in result or []:
            for line in line_group or []:
                if len(line) < 2:
                    continue
                text, conf = line[1]
                texts.append(str(text))
                confidences.append(float(conf))
        if not texts:
            return "", 0.0

        joined = " ".join(texts)
        confidence = max(confidences) if confidences else 0.0
        return joined, confidence

    def _variants(self, crop: np.ndarray) -> list[np.ndarray]:
        if not self.config.use_preprocess_variants:
            return [crop]
        import cv2

        variants: list[np.ndarray] = []
        for angle in self.config.rotate_angles:
            rotated = self._rotate(crop, angle)
            variants.append(rotated)
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if rotated.ndim == 3 else rotated
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))
            thresh = cv2.adaptiveThreshold(
                clahe,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )
            variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
        return variants

    @staticmethod
    def _rotate(image: np.ndarray, angle: int) -> np.ndarray:
        import cv2

        normalized = angle % 360
        if normalized == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if normalized == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if normalized == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def map_text_to_id(self, text: str) -> str | None:
        lowered = text.lower()
        for object_id, keywords in self.config.keyword_map.items():
            for keyword in keywords:
                if keyword.lower() in lowered:
                    return object_id
        return None

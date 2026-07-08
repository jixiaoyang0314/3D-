from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Frame:
    rgb: np.ndarray
    depth: np.ndarray | None = None
    timestamp: float = 0.0
    index: int = 0


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    frame_index: int = 0
    table_id: int = 1
    center_3d: tuple[float, float, float] | None = None
    depth_score: float = 0.0
    picture_like: bool = False
    depth_stats: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def adjusted_confidence(self) -> float:
        penalty = float(self.extra.get("picture_penalty", 0.05)) if self.picture_like else 1.0
        return max(0.0, min(1.0, self.confidence * penalty + self.depth_score))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(slots=True)
class Instance:
    class_name: str
    table_id: int
    score: float
    visible_frames: int
    best_xyxy: tuple[float, float, float, float]
    best_crop: np.ndarray | None = None
    crops: list[np.ndarray] = field(default_factory=list)
    center_3d: tuple[float, float, float] | None = None
    class_scores: dict[str, float] = field(default_factory=dict)
    classifier_scores: dict[str, float] = field(default_factory=dict)
    classifier_label: str = ""
    classifier_score: float = 0.0
    depth_validated: bool = False
    picture_like_votes: int = 0
    ocr_text: str = ""
    ocr_score: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OutputRow:
    object_id: str
    num: int
    table_id: int
    score: float = 0.0

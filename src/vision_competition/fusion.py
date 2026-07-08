from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RuntimeConfig
from .geometry import box_iou, center_distance, clip_box
from .types import Detection, Frame, Instance


@dataclass
class _Track:
    table_id: int
    detections: list[Detection] = field(default_factory=list)
    class_scores: dict[str, float] = field(default_factory=dict)
    best_confidence: float = 0.0
    best_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    best_crop: np.ndarray | None = None
    crops: list[np.ndarray] = field(default_factory=list)
    centers_3d: list[tuple[float, float, float]] = field(default_factory=list)
    picture_like_votes: int = 0
    depth_valid_votes: int = 0

    def add(self, det: Detection, frame: Frame, max_crops: int) -> None:
        self.detections.append(det)
        self.class_scores[det.class_name] = self.class_scores.get(det.class_name, 0.0) + det.adjusted_confidence
        if det.center_3d is not None:
            self.centers_3d.append(det.center_3d)
        if det.picture_like:
            self.picture_like_votes += 1
        if det.depth_score > 0:
            self.depth_valid_votes += 1
        h, w = frame.rgb.shape[:2]
        x1, y1, x2, y2 = clip_box(det.xyxy, w, h)
        crop = frame.rgb[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
        if crop is not None and len(self.crops) < max_crops:
            self.crops.append(crop)
        if det.adjusted_confidence > self.best_confidence:
            self.best_confidence = det.adjusted_confidence
            self.best_xyxy = det.xyxy
            self.best_crop = crop

    @property
    def visible_frames(self) -> int:
        return len({det.frame_index for det in self.detections})

    @property
    def last_box(self) -> tuple[float, float, float, float]:
        return self.detections[-1].xyxy

    @property
    def last_center_3d(self) -> tuple[float, float, float] | None:
        return self.centers_3d[-1] if self.centers_3d else None

    @property
    def median_center_3d(self) -> tuple[float, float, float] | None:
        if not self.centers_3d:
            return None
        center = np.median(np.asarray(self.centers_3d, dtype=np.float32), axis=0)
        return (float(center[0]), float(center[1]), float(center[2]))


class MultiFrameFusion:
    def __init__(self, runtime: RuntimeConfig) -> None:
        self.runtime = runtime
        self.tracks: list[_Track] = []

    def add_frame(self, frame: Frame, detections: list[Detection]) -> None:
        for det in detections:
            if det.adjusted_confidence < self.runtime.conf_floor:
                continue
            track = self._find_track(det)
            if track is None:
                track = _Track(table_id=det.table_id)
                self.tracks.append(track)
            track.add(det, frame, max_crops=self.runtime.max_crops_per_instance)

    def _find_track(self, det: Detection) -> _Track | None:
        best_score = 0.0
        best_track: _Track | None = None
        for track in self.tracks:
            if track.table_id != det.table_id:
                continue
            iou = box_iou(track.last_box, det.xyxy)
            same_class = det.class_name in track.class_scores
            if not same_class and iou < 0.55:
                continue
            score = 0.0
            if self.runtime.use_3d_fusion and det.center_3d is not None and track.last_center_3d is not None:
                det_center = np.asarray(det.center_3d, dtype=np.float32)
                track_center = np.asarray(track.last_center_3d, dtype=np.float32)
                distance_3d = float(np.linalg.norm(det_center - track_center))
                if distance_3d <= self.runtime.max_3d_match_distance_m:
                    score = 1.0 - distance_3d / max(self.runtime.max_3d_match_distance_m, 1e-6)
                elif same_class:
                    continue
            dist = center_distance(track.last_box, det.xyxy)
            norm = max(40.0, np.sqrt(max(det.area, 1.0)))
            dist_score = max(0.0, 1.0 - dist / (2.2 * norm))
            class_bonus = 0.15 if same_class else 0.0
            score = max(score, iou, dist_score) + class_bonus
            if score > best_score:
                best_score = score
                best_track = track
        return best_track if best_score >= 0.45 else None

    def finalize(self) -> list[Instance]:
        instances: list[Instance] = []
        for track in self.tracks:
            if not track.detections:
                continue
            class_name, class_score = max(track.class_scores.items(), key=lambda item: item[1])
            mean_conf = class_score / max(1, len(track.detections))
            stability_bonus = min(0.18, 0.04 * max(0, track.visible_frames - 1))
            score = max(track.best_confidence, mean_conf) + stability_bonus
            score = float(max(0.0, min(1.0, score)))

            # A track dominated by picture-like detections is risky, even if RGB confidence is high.
            if track.picture_like_votes > track.visible_frames * 0.5:
                score *= 0.25

            instances.append(
                Instance(
                    class_name=class_name,
                    table_id=track.table_id,
                    score=score,
                    visible_frames=track.visible_frames,
                    best_xyxy=track.best_xyxy,
                    best_crop=track.best_crop,
                    crops=list(track.crops),
                    center_3d=track.median_center_3d,
                    class_scores=dict(track.class_scores),
                    depth_validated=track.depth_valid_votes > 0,
                    picture_like_votes=track.picture_like_votes,
                    diagnostics={
                        "detections": len(track.detections),
                        "depth_valid_votes": track.depth_valid_votes,
                    },
                )
            )
        return instances


def draw_instances(image: np.ndarray, instances: list[Instance]) -> np.ndarray:
    import cv2

    canvas = image.copy()
    for inst in instances:
        x1, y1, x2, y2 = [int(round(v)) for v in inst.best_xyxy]
        color = (0, 220, 80) if inst.score >= 0.7 else (0, 180, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"T{inst.table_id} {inst.class_name} {inst.score:.2f}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return canvas

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CameraConfig, DepthConfig, RuntimeConfig
from .geometry import clip_box
from .types import Detection, Frame


@dataclass(slots=True)
class PlaneModel:
    normal: np.ndarray
    offset: float
    inlier_ratio: float

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.abs(points @ self.normal + self.offset)


def normalize_depth_to_meters(depth: np.ndarray) -> np.ndarray:
    depth_f = depth.astype(np.float32)
    valid = depth_f[np.isfinite(depth_f) & (depth_f > 0)]
    if valid.size == 0:
        return depth_f
    if float(np.nanmedian(valid)) > 20.0:
        depth_f = depth_f / 1000.0
    return depth_f


class DepthAnalyzer:
    def __init__(
        self,
        depth_config: DepthConfig,
        camera_config: CameraConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        self.depth_config = depth_config
        self.camera_config = camera_config
        self.runtime_config = runtime_config

    def enrich(self, frame: Frame, detections: list[Detection]) -> list[Detection]:
        h, w = frame.rgb.shape[:2]
        for det in detections:
            det.table_id = self.assign_table(det.xyxy, width=w, height=h)

        if not self.depth_config.enabled or frame.depth is None:
            return detections

        depth = normalize_depth_to_meters(frame.depth)
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size == 0:
            return detections

        table_depth = float(np.nanpercentile(valid, self.depth_config.table_depth_percentile))
        plane = self._estimate_table_plane(depth, width=w, height=h)
        for det in detections:
            self._attach_depth_stats(det, depth, table_depth, plane)
        return detections

    def assign_table(self, xyxy: tuple[float, float, float, float], width: int, height: int) -> int:
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        for idx, roi in enumerate(self.camera_config.table_rois, start=1):
            if len(roi) != 4:
                continue
            rx1, ry1, rx2, ry2 = roi
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return idx

        if self.runtime_config.table_count <= 1:
            return 1

        table_width = width / float(self.runtime_config.table_count)
        return max(1, min(self.runtime_config.table_count, int(cx // table_width) + 1))

    def _camera_intrinsics(self, width: int, height: int) -> tuple[float, float, float, float]:
        fx = self.camera_config.fx or float(max(width, height))
        fy = self.camera_config.fy or float(max(width, height))
        cx = self.camera_config.cx if self.camera_config.cx is not None else (width - 1) * 0.5
        cy = self.camera_config.cy if self.camera_config.cy is not None else (height - 1) * 0.5
        return float(fx), float(fy), float(cx), float(cy)

    def _points_from_depth(
        self,
        depth_patch: np.ndarray,
        x_offset: int,
        y_offset: int,
        full_width: int,
        full_height: int,
        mask_patch: np.ndarray | None = None,
    ) -> np.ndarray:
        stride = max(1, int(self.depth_config.point_sample_stride))
        patch = depth_patch[::stride, ::stride]
        valid = np.isfinite(patch) & (patch > 0)
        if mask_patch is not None:
            valid &= mask_patch[::stride, ::stride]
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        yy, xx = np.nonzero(valid)
        z = patch[yy, xx].astype(np.float32)
        yy = yy.astype(np.float32) * stride + y_offset
        xx = xx.astype(np.float32) * stride + x_offset
        fx, fy, cx, cy = self._camera_intrinsics(full_width, full_height)
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def _estimate_table_plane(self, depth: np.ndarray, width: int, height: int) -> PlaneModel | None:
        if not self.depth_config.ransac_plane_enabled:
            return None
        points = self._points_from_depth(depth, 0, 0, width, height)
        if points.shape[0] < 64:
            return None
        if points.shape[0] > self.depth_config.plane_max_points:
            rng = np.random.default_rng(7)
            indices = rng.choice(points.shape[0], self.depth_config.plane_max_points, replace=False)
            points = points[indices]

        rng = np.random.default_rng(17)
        best_inliers: np.ndarray | None = None
        best_normal: np.ndarray | None = None
        best_offset = 0.0
        threshold = float(self.depth_config.plane_threshold_m)

        for _ in range(max(1, int(self.depth_config.plane_iterations))):
            sample = points[rng.choice(points.shape[0], 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                continue
            normal = normal / norm
            offset = -float(normal @ sample[0])
            distances = np.abs(points @ normal + offset)
            inliers = distances < threshold
            if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
                best_inliers = inliers
                best_normal = normal
                best_offset = offset

        if best_inliers is None or best_normal is None or int(best_inliers.sum()) < 32:
            return None

        inlier_points = points[best_inliers]
        centroid = inlier_points.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        normal = vh[-1]
        normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
        offset = -float(normal @ centroid)
        inlier_ratio = float(best_inliers.mean())
        return PlaneModel(normal=normal.astype(np.float32), offset=offset, inlier_ratio=inlier_ratio)

    def _attach_depth_stats(
        self,
        det: Detection,
        depth: np.ndarray,
        table_depth: float,
        plane: PlaneModel | None,
    ) -> None:
        height, width = depth.shape[:2]
        x1, y1, x2, y2 = clip_box(det.xyxy, width, height)
        if x2 <= x1 or y2 <= y1:
            return

        patch = depth[y1:y2, x1:x2]
        mask_patch = None
        if det.mask is not None:
            mask = det.mask
            if mask.shape[:2] != depth.shape[:2]:
                import cv2

                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            mask_patch = mask[y1:y2, x1:x2] > 0.5
            values = patch[mask_patch]
        else:
            values = patch.reshape(-1)

        values = values[np.isfinite(values) & (values > 0)]
        total_pixels = max(1, (y2 - y1) * (x2 - x1))
        valid_ratio = float(values.size / total_pixels)
        if values.size < 16:
            det.depth_stats = {"valid_ratio": valid_ratio}
            return

        points = self._points_from_depth(patch, x1, y1, width, height, mask_patch=mask_patch)
        point_count = int(points.shape[0])
        if point_count > 0:
            center = np.nanmedian(points, axis=0)
            det.center_3d = (float(center[0]), float(center[1]), float(center[2]))

        p10 = float(np.nanpercentile(values, 10))
        p50 = float(np.nanpercentile(values, 50))
        p90 = float(np.nanpercentile(values, 90))
        thickness = abs(p90 - p10)
        plane_height_p50 = 0.0
        plane_height_p90 = 0.0
        plane_height_p95 = 0.0
        bbox_volume = 0.0

        if plane is not None and point_count >= self.depth_config.min_object_points:
            distances = plane.distances(points)
            plane_height_p50 = float(np.nanpercentile(distances, 50))
            plane_height_p90 = float(np.nanpercentile(distances, 90))
            plane_height_p95 = float(np.nanpercentile(distances, 95))
            object_height = plane_height_p90
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            extents = np.maximum(maxs - mins, 0.0)
            bbox_volume = float(extents[0] * extents[1] * extents[2])
        elif self.depth_config.object_closer_to_camera:
            object_height = max(0.0, table_depth - p50)
        else:
            object_height = max(0.0, p50 - table_depth)

        picture_like = (
            valid_ratio >= self.depth_config.min_valid_ratio
            and point_count >= self.depth_config.min_object_points
            and object_height <= self.depth_config.picture_height_m
            and thickness <= self.depth_config.picture_thickness_m
        )

        det.picture_like = picture_like
        if picture_like:
            det.extra["picture_penalty"] = self.depth_config.picture_penalty
        if not picture_like and valid_ratio >= self.depth_config.min_valid_ratio:
            det.depth_score = self.depth_config.depth_bonus

        det.depth_stats = {
            "valid_ratio": valid_ratio,
            "point_count": float(point_count),
            "table_depth_m": table_depth,
            "median_depth_m": p50,
            "height_from_table_m": object_height,
            "plane_height_p50_m": plane_height_p50,
            "plane_height_p90_m": plane_height_p90,
            "plane_height_p95_m": plane_height_p95,
            "depth_thickness_m": thickness,
            "bbox_volume_m3": bbox_volume,
            "plane_inlier_ratio": 0.0 if plane is None else plane.inlier_ratio,
        }

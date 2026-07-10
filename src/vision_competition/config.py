from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CompetitionConfig:
    team_prefix: str = "YOUR_UNIV"
    team_name: str = "YOUR_TEAM"
    round_id: int = 1
    result_dir: str = "~/Desktop/result"


@dataclass(slots=True)
class ModelConfig:
    yolo_weights: str = "weights/best.pt"
    yolo_weights_ensemble: list[str] = field(default_factory=list)
    classifier_weights: str | None = None
    classifier_enabled: bool = False
    classifier_min_confidence: float = 0.70
    classifier_weight: float = 0.35
    classifier_groups: list[list[str]] = field(default_factory=list)
    ocr_enabled: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cpu"
    imgsz: int = 1280
    multiscale_imgsz: list[int] = field(default_factory=list)
    wbf_iou: float = 0.55
    small_object_refine_enabled: bool = False
    small_object_refine_classes: list[str] = field(default_factory=list)
    small_object_refine_area_ratio: float = 0.04
    small_object_refine_max_detections: int = 8
    small_object_refine_crop_scale: float = 3.0
    small_object_refine_min_crop_side: int = 320
    small_object_refine_imgsz: int = 704
    small_object_refine_conf: float = 0.08
    small_object_refine_match_iou: float = 0.12
    frame_count: int = 24
    warmup_frames: int = 2
    max_runtime_sec: float = 55.0
    conf_floor: float = 0.12
    yolo_iou: float = 0.60
    stable_min_frames: int = 3
    high_conf: float = 0.78
    output_min_expected_score: float = 0.35
    class_thresholds: dict[str, float] = field(default_factory=dict)
    class_min_expected_scores: dict[str, float] = field(default_factory=dict)
    max_count_per_id: int = 5
    max_crops_per_instance: int = 6
    use_3d_fusion: bool = True
    max_3d_match_distance_m: float = 0.12
    table_count: int = 1


@dataclass(slots=True)
class CameraConfig:
    source: str = "replay"
    rgb_index: int = 0
    replay_dir: str = "replay"
    table_rois: list[list[int]] = field(default_factory=list)
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None


@dataclass(slots=True)
class DepthConfig:
    enabled: bool = True
    object_closer_to_camera: bool = True
    table_depth_percentile: float = 75.0
    ransac_plane_enabled: bool = True
    plane_max_points: int = 2500
    plane_iterations: int = 80
    plane_threshold_m: float = 0.012
    point_sample_stride: int = 3
    min_valid_ratio: float = 0.18
    min_object_points: int = 24
    picture_height_m: float = 0.018
    picture_thickness_m: float = 0.012
    picture_penalty: float = 0.05
    depth_bonus: float = 0.06


@dataclass(slots=True)
class OCRConfig:
    min_confidence: float = 0.82
    vote_min_count: int = 1
    use_preprocess_variants: bool = True
    rotate_angles: list[int] = field(default_factory=lambda: [0, 90, 180, 270])
    keyword_map: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class JudgeConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 9000
    timeout_sec: float = 1.0


@dataclass(slots=True)
class AppConfig:
    competition: CompetitionConfig = field(default_factory=CompetitionConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


def _build(cls: type[Any], data: dict[str, Any] | None) -> Any:
    data = data or {}
    valid = {field_name for field_name in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig(
        competition=_build(CompetitionConfig, raw.get("competition")),
        models=_build(ModelConfig, raw.get("models")),
        runtime=_build(RuntimeConfig, raw.get("runtime")),
        camera=_build(CameraConfig, raw.get("camera")),
        depth=_build(DepthConfig, raw.get("depth")),
        ocr=_build(OCRConfig, raw.get("ocr")),
        judge=_build(JudgeConfig, raw.get("judge")),
    )

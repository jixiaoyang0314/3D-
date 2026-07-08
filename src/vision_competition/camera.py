from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import CameraConfig
from .types import Frame


class CameraBase:
    def read(self) -> Frame | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ReplayCamera(CameraBase):
    """Reads rgb_*.jpg/png and optional depth_*.npy/png from a directory."""

    def __init__(self, replay_dir: str | Path):
        self.replay_dir = Path(replay_dir)
        patterns = ["rgb_*.jpg", "rgb_*.jpeg", "rgb_*.png", "*.jpg", "*.jpeg", "*.png"]
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(self.replay_dir.glob(pattern)))
            if files:
                break
        self.rgb_files = files
        self.index = 0

    def read(self) -> Frame | None:
        import cv2

        if self.index >= len(self.rgb_files):
            return None
        rgb_path = self.rgb_files[self.index]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            self.index += 1
            return self.read()

        stem = rgb_path.stem
        suffix = stem.split("_")[-1] if "_" in stem else f"{self.index:04d}"
        depth = self._read_depth(suffix, rgb.shape[:2])
        frame = Frame(rgb=rgb, depth=depth, timestamp=time.time(), index=self.index)
        self.index += 1
        return frame

    def _read_depth(self, suffix: str, shape: tuple[int, int]) -> np.ndarray | None:
        candidates = [
            self.replay_dir / f"depth_{suffix}.npy",
            self.replay_dir / f"depth_{suffix}.png",
            self.replay_dir / f"depth_{suffix}.tiff",
        ]
        for path in candidates:
            if not path.exists():
                continue
            if path.suffix == ".npy":
                depth = np.load(path)
            else:
                import cv2

                depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            if depth.shape[:2] != shape:
                import cv2

                depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return depth
        return None


class OpenCVCamera(CameraBase):
    """Fallback RGB-only camera for quick debugging."""

    def __init__(self, index: int = 0):
        import cv2

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        self.index = 0

    def read(self) -> Frame | None:
        ok, rgb = self.cap.read()
        if not ok:
            return None
        frame = Frame(rgb=rgb, depth=None, timestamp=time.time(), index=self.index)
        self.index += 1
        return frame

    def close(self) -> None:
        self.cap.release()


class OrbbecCamera(CameraBase):
    """Adapter placeholder for Astra Pro Plus.

    Fill this class with the SDK calls used on the OrangePi image. Keep the
    public read() method returning RGB in BGR OpenCV order and depth aligned to
    RGB with the same HxW resolution.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Connect the Orbbec/Astra SDK here before the real contest run.")


def create_camera(config: CameraConfig) -> CameraBase:
    source = config.source.lower()
    if source == "replay":
        return ReplayCamera(config.replay_dir)
    if source in {"webcam", "opencv"}:
        return OpenCVCamera(config.rgb_index)
    if source in {"orbbec", "astra"}:
        return OrbbecCamera()
    raise ValueError(f"Unsupported camera source: {config.source}")

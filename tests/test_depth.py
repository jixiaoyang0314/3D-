import unittest

import numpy as np

from vision_competition.config import CameraConfig, DepthConfig, RuntimeConfig
from vision_competition.depth import DepthAnalyzer
from vision_competition.types import Detection, Frame


class DepthTests(unittest.TestCase):
    def test_flat_patch_is_picture_like(self):
        depth = np.ones((80, 80), dtype=np.float32)
        frame = Frame(rgb=np.zeros((80, 80, 3), dtype=np.uint8), depth=depth)
        det = Detection(0, "CA001", 0.9, (20, 20, 60, 60))
        analyzer = DepthAnalyzer(DepthConfig(plane_iterations=20), CameraConfig(), RuntimeConfig())

        [out] = analyzer.enrich(frame, [det])

        self.assertTrue(out.picture_like)
        self.assertIsNotNone(out.center_3d)

    def test_raised_patch_is_not_picture_like(self):
        depth = np.ones((80, 80), dtype=np.float32)
        depth[25:55, 25:55] = 0.88
        frame = Frame(rgb=np.zeros((80, 80, 3), dtype=np.uint8), depth=depth)
        det = Detection(0, "CA001", 0.9, (20, 20, 60, 60))
        analyzer = DepthAnalyzer(DepthConfig(plane_iterations=20), CameraConfig(), RuntimeConfig())

        [out] = analyzer.enrich(frame, [det])

        self.assertFalse(out.picture_like)
        self.assertGreater(out.depth_stats["height_from_table_m"], 0.02)


if __name__ == "__main__":
    unittest.main()

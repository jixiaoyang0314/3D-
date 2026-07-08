import unittest

import numpy as np

from vision_competition.config import RuntimeConfig
from vision_competition.fusion import MultiFrameFusion
from vision_competition.types import Detection, Frame


class FusionTests(unittest.TestCase):
    def test_nearby_different_classes_do_not_merge_without_large_iou(self):
        fusion = MultiFrameFusion(RuntimeConfig())
        frame = Frame(rgb=np.zeros((100, 100, 3), dtype=np.uint8), index=1)

        fusion.add_frame(
            frame,
            [
                Detection(0, "bus", 0.90, (10, 10, 90, 90), frame_index=1),
                Detection(1, "tie", 0.80, (40, 40, 50, 55), frame_index=1),
            ],
        )

        instances = fusion.finalize()
        classes = sorted(instance.class_name for instance in instances)
        self.assertEqual(classes, ["bus", "tie"])

    def test_3d_centers_merge_same_object_even_when_2d_moves(self):
        fusion = MultiFrameFusion(RuntimeConfig(max_3d_match_distance_m=0.05))
        frame1 = Frame(rgb=np.zeros((100, 100, 3), dtype=np.uint8), index=1)
        frame2 = Frame(rgb=np.zeros((100, 100, 3), dtype=np.uint8), index=2)

        fusion.add_frame(
            frame1,
            [Detection(0, "CA001", 0.90, (10, 10, 30, 30), frame_index=1, center_3d=(0.1, 0.1, 1.0))],
        )
        fusion.add_frame(
            frame2,
            [Detection(0, "CA001", 0.92, (50, 50, 70, 70), frame_index=2, center_3d=(0.11, 0.1, 1.0))],
        )

        instances = fusion.finalize()
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].visible_frames, 2)
        self.assertIsNotNone(instances[0].center_3d)


if __name__ == "__main__":
    unittest.main()

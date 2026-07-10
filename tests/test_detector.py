import unittest

import numpy as np

from vision_competition.detector import YOLODetector
from vision_competition.types import Detection


class DetectorLogicTests(unittest.TestCase):
    def make_detector(self) -> YOLODetector:
        detector = YOLODetector.__new__(YOLODetector)
        detector.wbf_iou = 0.55
        detector.refine_enabled = True
        detector.refine_classes = {"CA002", "CB002", "CB003", "CC001", "CC003"}
        detector.refine_area_ratio = 0.04
        detector.refine_max_detections = 8
        detector.refine_crop_scale = 3.0
        detector.refine_min_crop_side = 320
        detector.refine_imgsz = 704
        detector.refine_conf = 0.08
        detector.refine_match_iou = 0.12
        detector.conf = 0.12
        detector.iou = 0.60
        detector.device = "cpu"
        detector.multiscale_imgsz = [672, 704]
        detector.names = {}
        return detector

    def test_weighted_boxes_fusion_merges_same_class(self) -> None:
        detector = self.make_detector()
        detections = [
            Detection(0, "CA002", 0.80, (10, 10, 30, 30), extra={"source": "imgsz_672"}),
            Detection(0, "CA002", 0.70, (11, 9, 31, 29), extra={"source": "imgsz_704"}),
            Detection(1, "CB002", 0.60, (60, 60, 90, 90), extra={"source": "imgsz_704"}),
        ]

        fused = detector._weighted_boxes_fusion(detections)

        self.assertEqual(len(fused), 2)
        ca002 = next(det for det in fused if det.class_name == "CA002")
        self.assertGreater(ca002.confidence, 0.80)
        self.assertEqual(ca002.extra["wbf_sources"], 2)
        self.assertAlmostEqual(ca002.xyxy[0], 10.47, places=1)
        self.assertAlmostEqual(ca002.xyxy[2], 30.47, places=1)

    def test_refine_small_objects_replaces_box_with_fused_local_match(self) -> None:
        detector = self.make_detector()
        original = Detection(0, "CA002", 0.60, (100, 100, 140, 140), extra={"source": "imgsz_704"})
        untouched = Detection(9, "CC004", 0.75, (300, 300, 420, 420), extra={"source": "imgsz_704"})

        detector._predict_crop = lambda crop, crop_origin, frame_index: [  # type: ignore[method-assign]
            Detection(0, "CA002", 0.72, (102, 101, 142, 141), frame_index=frame_index, extra={"source": "small_object_refine"})
        ]

        image = np.zeros((640, 640, 3), dtype=np.uint8)
        refined = detector._refine_small_objects(image, [original, untouched], frame_index=1)

        self.assertEqual(len(refined), 2)
        self.assertTrue(refined[0].extra["refined"])
        self.assertGreater(refined[0].confidence, original.confidence)
        self.assertAlmostEqual(refined[0].xyxy[0], 101.0, places=1)
        self.assertAlmostEqual(refined[0].xyxy[2], 141.0, places=1)
        self.assertFalse(refined[1].extra.get("refined", False))

    def test_best_refine_match_ignores_wrong_class(self) -> None:
        detector = self.make_detector()
        original = Detection(0, "CA002", 0.62, (100, 100, 140, 140))
        candidates = [
            Detection(1, "CB002", 0.95, (101, 101, 141, 141)),
            Detection(0, "CA002", 0.55, (200, 200, 240, 240)),
            Detection(0, "CA002", 0.70, (102, 102, 142, 142)),
        ]

        match = detector._best_refine_match(original, candidates)

        self.assertIsNotNone(match)
        self.assertEqual(match.class_name, "CA002")
        self.assertAlmostEqual(match.xyxy[0], 102.0, places=1)


if __name__ == "__main__":
    unittest.main()

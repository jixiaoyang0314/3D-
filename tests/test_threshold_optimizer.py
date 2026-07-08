import tempfile
import unittest
from pathlib import Path

from vision_competition.config import RuntimeConfig
from vision_competition.scorer import TruthRow
from vision_competition.threshold_optimizer import optimize_class_thresholds, write_threshold_yaml
from vision_competition.types import Instance


class ThresholdOptimizerTests(unittest.TestCase):
    def test_optimizer_prefers_threshold_that_removes_false_positive(self):
        samples = [
            (
                [Instance("CA001", 1, 0.70, 3, (0, 0, 1, 1))],
                [],
            ),
            (
                [Instance("CA001", 1, 0.90, 3, (0, 0, 1, 1))],
                [TruthRow("CA001", 1, 1)],
            ),
        ]
        runtime = RuntimeConfig(stable_min_frames=1, output_min_expected_score=0.0)
        thresholds = optimize_class_thresholds(samples, runtime, candidates=[0.1, 0.8])
        self.assertGreaterEqual(thresholds["CA001"], 0.8)

    def test_write_threshold_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_threshold_yaml(Path(tmp) / "thresholds.yaml", {"CA001": 0.5})
            self.assertIn("CA001", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

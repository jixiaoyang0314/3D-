import unittest

from vision_competition.config import RuntimeConfig
from vision_competition.decision import choose_count, decide_outputs
from vision_competition.types import Instance


class DecisionTests(unittest.TestCase):
    def test_conservative_when_uncertain(self):
        runtime = RuntimeConfig(output_min_expected_score=0.35, max_count_per_id=5)
        count, expected = choose_count([0.45], runtime)
        self.assertEqual(count, 0)
        self.assertLess(expected, 0.35)

    def test_outputs_stable_high_probabilities(self):
        runtime = RuntimeConfig(output_min_expected_score=0.35, max_count_per_id=5)
        count, expected = choose_count([0.95, 0.91], runtime)
        self.assertEqual(count, 2)
        self.assertGreater(expected, 2.0)

    def test_class_threshold_filters_low_score_instances(self):
        runtime = RuntimeConfig(
            stable_min_frames=1,
            class_thresholds={"CA001": 0.80},
            output_min_expected_score=0.0,
        )
        rows = decide_outputs(
            [
                Instance(
                    class_name="CA001",
                    table_id=1,
                    score=0.70,
                    visible_frames=3,
                    best_xyxy=(0, 0, 10, 10),
                )
            ],
            runtime,
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()

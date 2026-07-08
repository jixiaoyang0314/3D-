import unittest

from vision_competition.scorer import TruthRow, score_round
from vision_competition.types import OutputRow


class ScorerTests(unittest.TestCase):
    def test_exact_match(self):
        result = score_round(
            [TruthRow("CA003", 2, 1)],
            [OutputRow("CA003", 2, 1)],
        )
        self.assertEqual(result.measure_score, 3.0)

    def test_under_count_partial_score(self):
        result = score_round(
            [TruthRow("CA003", 4, 1)],
            [OutputRow("CA003", 2, 1)],
        )
        self.assertEqual(result.measure_score, 1.5)

    def test_false_id_penalty(self):
        result = score_round(
            [TruthRow("CA003", 1, 1)],
            [OutputRow("CA999", 1, 1)],
        )
        self.assertEqual(result.false_id_penalty, -3.0)


if __name__ == "__main__":
    unittest.main()


import tempfile
import unittest
from pathlib import Path

from vision_competition.recorder import parse_truth_file


class RecorderTests(unittest.TestCase):
    def test_parse_truth_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truth.txt"
            path.write_text("START\nCA001;2;1\nEND\n", encoding="utf-8")
            rows = parse_truth_file(path)
            self.assertEqual(rows[0].object_id, "CA001")
            self.assertEqual(rows[0].num, 2)
            self.assertEqual(rows[0].table_id, 1)


if __name__ == "__main__":
    unittest.main()

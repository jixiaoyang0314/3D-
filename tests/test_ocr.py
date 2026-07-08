import unittest

from vision_competition.config import OCRConfig
from vision_competition.ocr import UnknownOCR


class OCRTests(unittest.TestCase):
    def test_keyword_mapping(self):
        ocr = UnknownOCR(OCRConfig(keyword_map={"W001": ["高等数学", "math"]}), enabled=False)
        self.assertEqual(ocr.map_text_to_id("高等数学 上册"), "W001")
        self.assertEqual(ocr.map_text_to_id("basic math"), "W001")
        self.assertIsNone(ocr.map_text_to_id("unknown"))


if __name__ == "__main__":
    unittest.main()


import unittest

from vision_competition.classifier import ClassifierVerifier
from vision_competition.config import ModelConfig, RuntimeConfig
from vision_competition.types import Instance


class ClassifierTests(unittest.TestCase):
    def test_disabled_classifier_is_noop(self):
        verifier = ClassifierVerifier(ModelConfig(classifier_enabled=False), RuntimeConfig())
        instances = [Instance("CA001", 1, 0.9, 2, (0, 0, 1, 1))]
        self.assertIs(verifier.verify(instances), instances)
        self.assertEqual(instances[0].class_name, "CA001")


if __name__ == "__main__":
    unittest.main()


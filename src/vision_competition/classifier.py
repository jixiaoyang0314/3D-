from __future__ import annotations

from .config import ModelConfig, RuntimeConfig
from .types import Instance


class ClassifierVerifier:
    """Optional second-stage classifier for confusing classes.

    The implementation uses Ultralytics classification weights when configured.
    If classifier_enabled is false or classifier_weights is empty, verify() is a
    no-op so the main pipeline stays smooth during early development.
    """

    def __init__(self, model_config: ModelConfig, runtime_config: RuntimeConfig) -> None:
        self.config = model_config
        self.runtime = runtime_config
        self.model = None
        if not model_config.classifier_enabled or not model_config.classifier_weights:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics before enabling classifier verification.") from exc
        self.model = YOLO(str(model_config.classifier_weights))

    def _should_verify(self, class_name: str) -> bool:
        if not self.config.classifier_groups:
            return True
        return any(class_name in group for group in self.config.classifier_groups)

    def _allowed_group(self, class_name: str) -> set[str] | None:
        for group in self.config.classifier_groups:
            if class_name in group:
                return set(group)
        return None

    def verify(self, instances: list[Instance]) -> list[Instance]:
        if self.model is None:
            return instances

        for inst in instances:
            if inst.best_crop is None or not self._should_verify(inst.class_name):
                continue
            result = self.model.predict(
                source=inst.best_crop,
                device=self.runtime.device,
                verbose=False,
            )[0]
            probs = getattr(result, "probs", None)
            if probs is None:
                continue

            names = result.names
            top1 = int(probs.top1)
            top1_conf = float(probs.top1conf)
            label = str(names.get(top1, top1))
            inst.classifier_label = label
            inst.classifier_score = top1_conf
            inst.classifier_scores[label] = top1_conf

            allowed = self._allowed_group(inst.class_name)
            if allowed is not None and label not in allowed:
                inst.score *= max(0.35, 1.0 - self.config.classifier_weight)
                continue

            if top1_conf >= self.config.classifier_min_confidence:
                if label != inst.class_name:
                    inst.class_name = label
                inst.score = min(1.0, inst.score * (1.0 - self.config.classifier_weight) + top1_conf * self.config.classifier_weight)

        return instances


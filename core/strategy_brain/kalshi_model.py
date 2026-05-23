"""
Probability model helpers for Kalshi strategy decisions.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from core.strategy_brain.kalshi_features import KalshiFeatureSnapshot


@dataclass
class KalshiModelPrediction:
    predicted_prob: float
    confidence: float
    model_version: str
    fallback_used: bool
    reason: str
    eval_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationBucket:
    bucket: str
    predicted_avg: float
    actual_rate: float
    sample_count: int


class ProbabilityModel(Protocol):
    def predict(
        self,
        snapshot: KalshiFeatureSnapshot,
        fallback_prob: float,
        fallback_confidence: float,
    ) -> KalshiModelPrediction:
        ...


def _clamp_probability(value: float) -> float:
    return max(0.02, min(0.98, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class KalshiProbabilityModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        eval_path: Optional[str] = None,
    ) -> None:
        self.model_path = Path(
            model_path or os.getenv("KALSHI_MODEL_PATH", "kalshi_probability_model.json")
        )
        default_eval = self.model_path.with_name("kalshi_model_eval.json")
        self.eval_path = Path(eval_path or os.getenv("KALSHI_MODEL_EVAL_PATH", str(default_eval)))
        self.model = self._load_json(self.model_path)
        self.eval_artifact = self._load_json(self.eval_path)

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _usable_model(self) -> bool:
        if not self.model:
            return False
        if self.model.get("type") != "logistic":
            return False
        return isinstance(self.model.get("weights"), dict)

    def _eval_metrics(self) -> Dict[str, float]:
        if not self.eval_artifact:
            return {}
        metrics: Dict[str, float] = {}
        for section in ("validation", "out_of_sample", "train"):
            values = self.eval_artifact.get(section)
            if not isinstance(values, dict):
                continue
            for key in ("brier_score", "log_loss", "directional_accuracy", "samples"):
                value = values.get(key)
                if isinstance(value, (int, float)):
                    metrics[f"{section}_{key}"] = float(value)
        return metrics

    def predict(
        self,
        snapshot: KalshiFeatureSnapshot,
        fallback_prob: float,
        fallback_confidence: float,
    ) -> KalshiModelPrediction:
        if not self._usable_model():
            return KalshiModelPrediction(
                predicted_prob=_clamp_probability(fallback_prob),
                confidence=max(0.0, min(1.0, fallback_confidence)),
                model_version="fusion-fallback",
                fallback_used=True,
                reason="missing_model",
                eval_metrics=self._eval_metrics(),
            )

        weights = self.model.get("weights", {})
        means = self.model.get("feature_means", {})
        scales = self.model.get("feature_scales", {})
        intercept = float(self.model.get("intercept", 0.0))
        score = intercept
        for name, weight in weights.items():
            raw_value = float(snapshot.features.get(str(name), 0.0))
            mean = float(means.get(str(name), 0.0)) if isinstance(means, dict) else 0.0
            scale = float(scales.get(str(name), 1.0)) if isinstance(scales, dict) else 1.0
            value = (raw_value - mean) / scale if scale else raw_value
            score += float(weight) * value
        probability = _clamp_probability(_sigmoid(score))
        confidence = min(1.0, max(0.0, abs(probability - 0.5) * 2.0))
        return KalshiModelPrediction(
            predicted_prob=probability,
            confidence=confidence,
            model_version=str(self.model.get("version", "logistic-v1")),
            fallback_used=False,
            reason="ok",
            eval_metrics=self._eval_metrics(),
        )

    def calibration_check(self, prediction_rows: List[Dict[str, Any]]) -> List[CalibrationBucket]:
        buckets: List[List[float]] = [[] for _ in range(10)]
        actuals: List[List[int]] = [[] for _ in range(10)]
        for row in prediction_rows:
            predicted = row.get("predicted_prob")
            actual = row.get("actual")
            if not isinstance(predicted, (int, float)) or actual not in {0, 1, False, True}:
                continue
            bucket_idx = min(9, max(0, int(float(predicted) * 10)))
            buckets[bucket_idx].append(float(predicted))
            actuals[bucket_idx].append(int(actual))

        summary: List[CalibrationBucket] = []
        for index, values in enumerate(buckets):
            if not values:
                continue
            start = index * 10
            end = start + 10
            summary.append(
                CalibrationBucket(
                    bucket=f"{start}-{end}%",
                    predicted_avg=sum(values) / len(values),
                    actual_rate=sum(actuals[index]) / len(actuals[index]),
                    sample_count=len(values),
                )
            )
        return summary


def load_probability_model(
    model_type: Optional[str] = None,
    *,
    model_path: Optional[str] = None,
    eval_path: Optional[str] = None,
):
    selected = (model_type or os.getenv("KALSHI_MODEL_TYPE", "logistic")).lower()
    if selected == "gbm":
        from core.strategy_brain.kalshi_gbm_model import KalshiGBMModel

        return KalshiGBMModel(model_path=model_path, eval_path=eval_path)
    return KalshiProbabilityModel(model_path=model_path, eval_path=eval_path)

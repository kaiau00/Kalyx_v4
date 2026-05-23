"""
Gradient boosting probability model for Kalshi decisions.
"""
from __future__ import annotations

import base64
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:  # pragma: no cover - optional dependency
    GradientBoostingClassifier = None

from core.strategy_brain.kalshi_features import KalshiFeatureSnapshot
from core.strategy_brain.kalshi_model import KalshiModelPrediction


def _clamp_probability(value: float) -> float:
    return max(0.02, min(0.98, value))


class KalshiGBMModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        eval_path: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path or os.getenv("KALSHI_GBM_MODEL_PATH", "kalshi_gbm_model.json"))
        default_eval = self.model_path.with_name("kalshi_gbm_eval.json")
        self.eval_path = Path(eval_path or os.getenv("KALSHI_GBM_EVAL_PATH", str(default_eval)))
        self.model: Optional[Any] = None
        self.feature_names: list[str] = []
        self.eval_artifact = self._load_json(self.eval_path)
        self._load_model()

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _load_model(self) -> bool:
        data = self._load_json(self.model_path)
        if not data or data.get("type") != "gbm":
            return False
        payload = data.get("pickle_b64")
        if not isinstance(payload, str):
            return False
        try:
            self.model = pickle.loads(base64.b64decode(payload.encode("ascii")))
            self.feature_names = [str(name) for name in data.get("feature_names", [])]
            return True
        except Exception:
            self.model = None
            return False

    def save(self) -> None:
        if self.model is None:
            return
        payload = {
            "type": "gbm",
            "version": "gbm-v1",
            "feature_names": self.feature_names,
            "pickle_b64": base64.b64encode(pickle.dumps(self.model)).decode("ascii"),
        }
        self.model_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def predict(
        self,
        snapshot: KalshiFeatureSnapshot,
        fallback_prob: float,
        fallback_confidence: float,
    ) -> KalshiModelPrediction:
        if self.model is None:
            return KalshiModelPrediction(
                predicted_prob=_clamp_probability(fallback_prob),
                confidence=max(0.0, min(1.0, fallback_confidence)),
                model_version="gbm-fallback",
                fallback_used=True,
                reason="missing_model",
                eval_metrics={},
            )
        vector = [float(snapshot.features.get(name, 0.0) or 0.0) for name in self.feature_names]
        try:
            probability = float(self.model.predict_proba([vector])[0][1])
        except Exception:
            return KalshiModelPrediction(
                predicted_prob=_clamp_probability(fallback_prob),
                confidence=max(0.0, min(1.0, fallback_confidence)),
                model_version="gbm-error",
                fallback_used=True,
                reason="model_error",
                eval_metrics={},
            )
        probability = _clamp_probability(probability)
        return KalshiModelPrediction(
            predicted_prob=probability,
            confidence=min(1.0, abs(probability - 0.5) * 2.0),
            model_version="gbm-v1",
            fallback_used=False,
            reason="ok",
            eval_metrics={},
        )

    def fit(self, rows: list[list[float]], labels: list[int], feature_names: list[str]) -> None:
        if GradientBoostingClassifier is None:  # pragma: no cover - optional dependency
            raise RuntimeError("scikit-learn is required to train the GBM model")
        self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(rows, labels)
        self.feature_names = list(feature_names)

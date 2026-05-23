import json

from core.strategy_brain.kalshi_features import KalshiFeatureSnapshot
from core.strategy_brain.kalshi_model import KalshiProbabilityModel


def test_probability_model_falls_back_without_artifact(tmp_path):
    model = KalshiProbabilityModel(model_path=str(tmp_path / "missing.json"))
    prediction = model.predict(KalshiFeatureSnapshot(features={}, metadata={}), fallback_prob=0.61, fallback_confidence=0.4)

    assert prediction.predicted_prob == 0.61
    assert prediction.confidence == 0.4
    assert prediction.fallback_used


def test_probability_model_uses_logistic_artifact(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "type": "logistic",
                "version": "test-v1",
                "intercept": 0,
                "weights": {"edge": 4},
                "feature_means": {"edge": 0},
                "feature_scales": {"edge": 1},
            }
        )
    )
    model = KalshiProbabilityModel(model_path=str(path))
    prediction = model.predict(
        KalshiFeatureSnapshot(features={"edge": 1.0}, metadata={}),
        fallback_prob=0.5,
        fallback_confidence=0,
    )

    assert prediction.predicted_prob > 0.9
    assert prediction.model_version == "test-v1"
    assert not prediction.fallback_used


def test_probability_model_loads_eval_metrics(tmp_path):
    model_path = tmp_path / "model.json"
    eval_path = tmp_path / "eval.json"
    model_path.write_text(
        json.dumps(
            {
                "type": "logistic",
                "version": "test-v2",
                "intercept": 0,
                "weights": {"edge": 2},
                "feature_means": {"edge": 0},
                "feature_scales": {"edge": 1},
            }
        )
    )
    eval_path.write_text(
        json.dumps(
            {
                "validation": {"brier_score": 0.12, "directional_accuracy": 0.63},
                "out_of_sample": {"log_loss": 0.55},
            }
        )
    )
    model = KalshiProbabilityModel(model_path=str(model_path), eval_path=str(eval_path))
    prediction = model.predict(
        KalshiFeatureSnapshot(features={"edge": 1.0}, metadata={}),
        fallback_prob=0.5,
        fallback_confidence=0,
    )

    assert prediction.eval_metrics["validation_brier_score"] == 0.12
    assert prediction.eval_metrics["out_of_sample_log_loss"] == 0.55


def test_calibration_check_buckets_predictions(tmp_path):
    model = KalshiProbabilityModel(model_path=str(tmp_path / "missing.json"))
    summary = model.calibration_check(
        [
            {"predicted_prob": 0.12, "actual": 0},
            {"predicted_prob": 0.18, "actual": 0},
            {"predicted_prob": 0.83, "actual": 1},
        ]
    )

    assert len(summary) == 2
    assert summary[0].bucket == "10-20%"
    assert summary[1].bucket == "80-90%"

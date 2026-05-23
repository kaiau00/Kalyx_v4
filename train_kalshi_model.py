#!/usr/bin/env python3
"""
Train a logistic probability model from settled Kalshi strategy logs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple


DEFAULT_FEATURES = [
    "spot_return_30s",
    "spot_return_1m",
    "spot_return_3m",
    "spot_return_5m",
    "realized_vol_3m",
    "realized_vol_5m",
    "realized_vol_15m",
    "volatility_acceleration",
    "wick_pressure_3m",
    "minutes_to_expiry",
    "spread_cents",
    "top_depth",
    "book_imbalance",
    "market_probability",
    "raw_fusion_probability",
    "market_probability_gap",
    "funding_rate",
    "open_interest",
    "funding_oi_divergence",
    "fear_greed",
]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _result(record: Dict[str, Any]) -> str | None:
    value = record.get("result") or record.get("settlement_result")
    return value if value in {"yes", "no"} else None


def _load_rows(path: Path, features: list[str]) -> Tuple[List[List[float]], List[int]]:
    rows: List[List[float]] = []
    labels: List[int] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result = _result(record)
        feature_map = record.get("features") or {}
        if result not in {"yes", "no"} or not feature_map:
            continue
        rows.append([float(feature_map.get(name, 0.0) or 0.0) for name in features])
        labels.append(1 if result == "yes" else 0)
    return rows, labels


def _standardize(rows: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    columns = list(zip(*rows))
    means = [mean(column) for column in columns]
    scales = [pstdev(column) or 1.0 for column in columns]
    normalized = [
        [(value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in rows
    ]
    return normalized, means, scales


def _apply_standardization(rows: List[List[float]], means: List[float], scales: List[float]) -> List[List[float]]:
    return [
        [(value - means[index]) / scales[index] if scales[index] else value for index, value in enumerate(row)]
        for row in rows
    ]


def _metrics(rows: List[List[float]], labels: List[int], intercept: float, weights: List[float]) -> Dict[str, float]:
    brier = 0.0
    log_loss = 0.0
    correct = 0
    predictions = []
    for row, label in zip(rows, labels):
        prediction = max(0.001, min(0.999, _sigmoid(intercept + sum(weight * value for weight, value in zip(weights, row)))))
        predictions.append(prediction)
        brier += (prediction - label) ** 2
        log_loss += -(label * math.log(prediction) + (1 - label) * math.log(1 - prediction))
        correct += int((prediction >= 0.5) == bool(label))
    count = len(rows)
    return {
        "samples": count,
        "brier_score": brier / count if count else 0.0,
        "log_loss": log_loss / count if count else 0.0,
        "directional_accuracy": correct / count if count else 0.0,
        "predicted_probability_mean": sum(predictions) / count if count else 0.0,
    }


def _train(
    train_rows: List[List[float]],
    train_labels: List[int],
    val_rows: List[List[float]],
    val_labels: List[int],
    epochs: int,
    lr: float,
    l2: float,
    patience: int = 100,
) -> Tuple[float, List[float], Dict[str, float]]:
    weights = [0.0 for _ in train_rows[0]]
    intercept = 0.0
    count = float(len(train_rows))

    best_val_loss = float("inf")
    best_intercept = intercept
    best_weights = list(weights)
    epochs_no_improve = 0

    for epoch in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for row, label in zip(train_rows, train_labels):
            prediction = _sigmoid(intercept + sum(w * v for w, v in zip(weights, row)))
            error = prediction - label
            grad_b += error
            for idx, val in enumerate(row):
                grad_w[idx] += error * val
        intercept -= lr * (grad_b / count)
        for idx, w in enumerate(weights):
            weights[idx] -= lr * ((grad_w[idx] / count) + (l2 * w))

        val_loss = 0.0
        for row, label in zip(val_rows, val_labels):
            pred = max(0.001, min(0.999, _sigmoid(intercept + sum(w * v for w, v in zip(weights, row)))))
            val_loss -= label * math.log(pred) + (1 - label) * math.log(1 - pred)
        val_loss /= len(val_rows) if val_rows else 1.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_intercept = intercept
            best_weights = list(weights)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                return best_intercept, best_weights, {
                    "val_loss": best_val_loss,
                    "stopped_at": epoch + 1,
                    "patience": patience,
                    "early_stopped": 1.0,
                }

    return best_intercept, best_weights, {
        "val_loss": best_val_loss,
        "stopped_at": epochs,
        "patience": patience,
        "early_stopped": 0.0,
    }


def _split_rows(
    rows: List[List[float]],
    labels: List[int],
    train_split: float,
    out_of_sample: float,
) -> Dict[str, Tuple[List[List[float]], List[int]]]:
    total = len(rows)
    oos_start = total
    if out_of_sample > 0:
        oos_size = max(1, int(total * out_of_sample))
        oos_start = max(1, total - oos_size)

    dev_rows = rows[:oos_start]
    dev_labels = labels[:oos_start]
    split_idx = max(1, min(len(dev_rows) - 1, int(len(dev_rows) * train_split)))
    return {
        "train": (dev_rows[:split_idx], dev_labels[:split_idx]),
        "validation": (dev_rows[split_idx:], dev_labels[split_idx:]),
        "out_of_sample": (rows[oos_start:], labels[oos_start:]) if oos_start < total else ([], []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a logistic Kalshi probability model.")
    parser.add_argument("--log", default="kalshi_trades.jsonl", help="Settled strategy log path.")
    parser.add_argument("--output", default="kalshi_probability_model.json", help="Model output path.")
    parser.add_argument("--eval-output", default="kalshi_model_eval.json", help="Evaluation JSON output path.")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--train-split", type=float, default=0.7, help="Chronological training split inside the development set.")
    parser.add_argument("--out-of-sample", type=float, default=0.3, help="Held-out tail fraction for out-of-sample evaluation.")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum settled rows required to train.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    args = parser.parse_args()

    rows, labels = _load_rows(Path(args.log), DEFAULT_FEATURES)
    if len(rows) < args.min_samples:
        raise SystemExit(f"Need at least {args.min_samples} settled rows with feature snapshots to train a model.")

    splits = _split_rows(rows, labels, args.train_split, args.out_of_sample)
    train_rows, train_labels = splits["train"]
    val_rows, val_labels = splits["validation"]
    oos_rows, oos_labels = splits["out_of_sample"]
    if not train_rows or not val_rows:
        raise SystemExit("Chronological split left an empty train or validation set.")

    train_normalized, means, scales = _standardize(train_rows)
    val_normalized = _apply_standardization(val_rows, means, scales)
    oos_normalized = _apply_standardization(oos_rows, means, scales) if oos_rows else []

    intercept, weights, train_info = _train(
        train_normalized,
        train_labels,
        val_normalized,
        val_labels,
        args.epochs,
        args.learning_rate,
        args.l2,
        patience=args.patience,
    )

    train_metrics = _metrics(train_normalized, train_labels, intercept, weights)
    val_metrics = _metrics(val_normalized, val_labels, intercept, weights)
    oos_metrics = _metrics(oos_normalized, oos_labels, intercept, weights) if oos_normalized else {
        "samples": 0,
        "brier_score": 0.0,
        "log_loss": 0.0,
        "directional_accuracy": 0.0,
        "predicted_probability_mean": 0.0,
    }

    model = {
        "type": "logistic",
        "version": "logistic-v2",
        "intercept": intercept,
        "weights": dict(zip(DEFAULT_FEATURES, weights)),
        "feature_means": dict(zip(DEFAULT_FEATURES, means)),
        "feature_scales": dict(zip(DEFAULT_FEATURES, scales)),
    }
    evaluation = {
        "train": train_metrics,
        "validation": val_metrics,
        "out_of_sample": oos_metrics,
        "train_info": train_info,
        "train_split": args.train_split,
        "out_of_sample_fraction": args.out_of_sample,
        "min_samples": args.min_samples,
    }

    Path(args.output).write_text(json.dumps(model, indent=2, sort_keys=True))
    Path(args.eval_output).write_text(json.dumps(evaluation, indent=2, sort_keys=True))
    print(json.dumps(evaluation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

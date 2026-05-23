#!/usr/bin/env python3
"""
Compare fusion-only, model-assisted, and market-implied baselines on historical windows.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtesting.historical_data import HistoricalDataLoader
from core.strategy_brain.kalshi_features import build_feature_snapshot
from core.strategy_brain.kalshi_fusion import KalshiSignalFusion
from core.strategy_brain.kalshi_indicators import compute_indicators, kalshi_market_signals, normalize_indicators
from core.strategy_brain.kalshi_model import KalshiProbabilityModel, load_probability_model
from execution.kalshi_api import ContractRef, OrderBookLevel, OrderBookSnapshot


def _contract_from_window(window):
    return ContractRef(
        ticker=f"SIM-{window.window_id}",
        event_ticker=None,
        title="Simulated contract",
        open_time=window.open_time,
        close_time=window.close_time,
        status="open",
    )


def _book_from_snapshot(snapshot):
    return OrderBookSnapshot(
        ticker=snapshot["ticker"],
        yes_bids=[OrderBookLevel(price=Decimal(str(row["price"])), quantity=Decimal(str(row["quantity"]))) for row in snapshot["yes_bids"]],
        no_bids=[OrderBookLevel(price=Decimal(str(row["price"])), quantity=Decimal(str(row["quantity"]))) for row in snapshot["no_bids"]],
        timestamp=snapshot["timestamp"],
    )


def _accuracy(probabilities, actual_yes):
    if not probabilities:
        return 0.0
    correct = sum(int((prob >= 0.5) == actual) for prob, actual in zip(probabilities, actual_yes))
    return correct / len(probabilities)


async def run_comparison(args):
    loader = HistoricalDataLoader(regime=args.regime)
    windows = await loader.load_windows(args.start, args.end)
    await loader.close()

    fusion = KalshiSignalFusion(state_path="/tmp/kalshi_backtest_model_signal_state.json", min_samples=1)
    logistic = KalshiProbabilityModel(model_path=args.model, eval_path=args.eval)
    selected_model = load_probability_model(model_type=args.model_type, model_path=args.model, eval_path=args.eval)

    fusion_probs = []
    model_probs = []
    market_probs = []
    actual_yes = []

    for window in windows:
        orderbook_dict = loader.generate_synthetic_orderbook(window, market_probability=0.5)
        orderbook = _book_from_snapshot(orderbook_dict)
        candles = window.candles
        signals = normalize_indicators(compute_indicators(candles), candles)
        signals.extend(kalshi_market_signals(orderbook))
        fused = fusion.fuse(signals)
        snapshot = build_feature_snapshot(
            candles=candles,
            orderbook=orderbook,
            contract=_contract_from_window(window),
            now=window.open_time,
            raw_fusion_probability=fused.predicted_prob,
            asset=args.asset,
            extra_features={signal.name: signal.value for signal in signals},
        )
        model_prediction = selected_model.predict(snapshot, fused.predicted_prob, fused.confidence)
        logistic_prediction = logistic.predict(snapshot, fused.predicted_prob, fused.confidence)
        fusion_probs.append(logistic_prediction.predicted_prob if args.model_type == "logistic" and logistic_prediction.fallback_used is False else fused.predicted_prob)
        model_probs.append(model_prediction.predicted_prob)
        market_probs.append(snapshot.features["market_probability"])
        actual_yes.append(window.ground_truth == "yes")

    report = {
        "samples": len(actual_yes),
        "fusion_only_accuracy": _accuracy(fusion_probs, actual_yes),
        "model_accuracy": _accuracy(model_probs, actual_yes),
        "market_implied_accuracy": _accuracy(market_probs, actual_yes),
        "model_type": args.model_type,
        "asset": args.asset,
        "regime": args.regime,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="Backtest model variants against historical windows.")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--regime", default="sideways")
    parser.add_argument("--start", default="2024-10-01T00:00:00+00:00")
    parser.add_argument("--end", default="2024-10-05T00:00:00+00:00")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "gbm"])
    parser.add_argument("--model", default="kalshi_probability_model.json")
    parser.add_argument("--eval", default="kalshi_model_eval.json")
    args = parser.parse_args()
    args.start = datetime.fromisoformat(args.start)
    args.end = datetime.fromisoformat(args.end)
    asyncio.run(run_comparison(args))


if __name__ == "__main__":
    main()

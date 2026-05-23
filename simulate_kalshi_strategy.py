#!/usr/bin/env python3
"""
Replay Kalshi strategy decisions from historical candles and orderbook snapshots.
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

from core.strategy_brain.kalshi_features import build_feature_snapshot
from core.strategy_brain.kalshi_fusion import KalshiSignalFusion
from core.strategy_brain.kalshi_indicators import (
    Candle,
    compute_indicators,
    kalshi_market_signals,
    normalize_indicators,
)
from core.strategy_brain.kalshi_model import KalshiProbabilityModel
from execution.kalshi_api import ContractRef, OrderBookLevel, OrderBookSnapshot
from execution.kalshi_risk import KalshiRiskConfig, build_trade_intent


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _candle(row: Dict[str, Any]) -> Candle:
    return Candle(
        open=_decimal(row["open"]),
        high=_decimal(row["high"]),
        low=_decimal(row["low"]),
        close=_decimal(row["close"]),
        volume=_decimal(row.get("volume", "0")),
    )


def _book(row: Dict[str, Any]) -> OrderBookSnapshot:
    raw_book = row["orderbook"]
    return OrderBookSnapshot(
        ticker=row.get("ticker", "SIM"),
        yes_bids=[OrderBookLevel(_decimal(p), _decimal(q)) for p, q in raw_book.get("yes", [])],
        no_bids=[OrderBookLevel(_decimal(p), _decimal(q)) for p, q in raw_book.get("no", [])],
        timestamp=datetime.now(timezone.utc),
    )


def _contract(row: Dict[str, Any], now: datetime) -> ContractRef:
    minutes_to_expiry = float(row.get("minutes_to_expiry", 7.5))
    return ContractRef(
        ticker=row.get("ticker", "SIM"),
        event_ticker=row.get("event_ticker"),
        title=row.get("title", "Simulated BTC 15m"),
        open_time=now - timedelta(minutes=15 - minutes_to_expiry),
        close_time=now + timedelta(minutes=minutes_to_expiry),
        status="open",
    )


def _bucket(value: float, edges: list[float]) -> str:
    for edge in edges:
        if value <= edge:
            return f"<= {edge:g}"
    return f"> {edges[-1]:g}"


def run_simulation(path: Path, model_path: str, slippage_cents: Decimal) -> Dict[str, Any]:
    fusion = KalshiSignalFusion(state_path="/tmp/kalshi_sim_signal_state.json", min_samples=1)
    model = KalshiProbabilityModel(model_path=model_path)
    config = KalshiRiskConfig()
    stats: Dict[str, Any] = {
        "snapshots": 0,
        "trades": 0,
        "skipped": 0,
        "correct": 0,
        "gross_pnl": Decimal("0"),
        "net_pnl": Decimal("0"),
        "max_drawdown": Decimal("0"),
        "skip_reasons": {},
        "pnl_by_edge_bucket": {},
        "pnl_by_expiry_bucket": {},
    }
    equity = Decimal("0")
    peak = Decimal("0")

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        now = datetime.fromisoformat(row.get("timestamp", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00"))
        candles = [_candle(candle) for candle in row["candles"]]
        orderbook = _book(row)
        contract = _contract(row, now)
        signals = normalize_indicators(compute_indicators(candles), candles)
        signals.extend(kalshi_market_signals(orderbook))
        fused = fusion.fuse(signals)
        snapshot = build_feature_snapshot(
            candles=candles,
            orderbook=orderbook,
            contract=contract,
            now=now,
            raw_fusion_probability=fused.predicted_prob,
        )
        prediction = model.predict(snapshot, fallback_prob=fused.predicted_prob, fallback_confidence=fused.confidence)
        intent = build_trade_intent(
            predicted_prob=prediction.predicted_prob,
            yes_ask=orderbook.best_yes_ask,
            no_ask=orderbook.best_no_ask,
            config=config,
            market_probability=snapshot.features["market_probability"],
            spread_cents=Decimal(str(snapshot.features["spread_cents"])),
            top_depth=Decimal(str(snapshot.features["top_depth"])),
            model_confidence=prediction.confidence,
            seconds_to_expiry=snapshot.features["seconds_to_expiry"],
        )
        yes_won = row.get("result") == "yes"
        stats["snapshots"] += 1

        if not intent.should_trade:
            stats["skipped"] += 1
            stats["skip_reasons"][intent.reason] = stats["skip_reasons"].get(intent.reason, 0) + 1
            fusion.update_from_outcome(signals, yes_won=yes_won)
            continue

        stats["trades"] += 1
        stats["correct"] += int((intent.side == "yes") == yes_won)
        entry_price = intent.limit_price + (slippage_cents / Decimal("100"))
        won = (intent.side == "yes") == yes_won
        gross = (Decimal("1") - entry_price) if won else -entry_price
        net = gross - config.fee_per_contract
        trade_pnl = net * Decimal(intent.quantity)
        stats["gross_pnl"] += gross * Decimal(intent.quantity)
        stats["net_pnl"] += trade_pnl
        equity += trade_pnl
        peak = max(peak, equity)
        stats["max_drawdown"] = min(stats["max_drawdown"], equity - peak)
        edge_bucket = _bucket(float(intent.edge_after_fees), [0.03, 0.05, 0.08, 0.12])
        expiry_bucket = _bucket(float(snapshot.features["minutes_to_expiry"]), [1, 3, 5, 10])
        stats["pnl_by_edge_bucket"][edge_bucket] = stats["pnl_by_edge_bucket"].get(edge_bucket, Decimal("0")) + trade_pnl
        stats["pnl_by_expiry_bucket"][expiry_bucket] = stats["pnl_by_expiry_bucket"].get(expiry_bucket, Decimal("0")) + trade_pnl
        fusion.update_from_outcome(signals, yes_won=yes_won)

    return stats


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Kalshi strategy on JSONL snapshots.")
    parser.add_argument("path", help="Path to JSONL snapshots.")
    parser.add_argument("--model", default="kalshi_probability_model.json", help="Optional model JSON path.")
    parser.add_argument("--slippage-cents", default="1", help="Slippage buffer per contract in cents.")
    args = parser.parse_args()

    stats = run_simulation(Path(args.path), args.model, Decimal(str(args.slippage_cents)))
    win_rate = stats["correct"] / stats["trades"] if stats["trades"] else 0
    stats["win_rate"] = win_rate
    print(json.dumps(stats, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()

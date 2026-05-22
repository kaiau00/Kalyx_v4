"""
Feature extraction for Kalshi crypto 15-minute markets.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from statistics import pstdev
from typing import Dict, Optional

from core.strategy_brain.kalshi_indicators import Candle, orderbook_imbalance
from execution.kalshi_api import ContractRef, OrderBookSnapshot


@dataclass
class KalshiFeatureSnapshot:
    features: Dict[str, float]
    metadata: Dict[str, str | float | int | None]


def _safe_float(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _return_over(closes: list[float], periods: int) -> float:
    if len(closes) <= periods or closes[-periods - 1] <= 0:
        return 0.0
    return (closes[-1] / closes[-periods - 1]) - 1.0


def _realized_vol(closes: list[float], periods: int) -> float:
    if len(closes) <= periods:
        return 0.0
    returns = []
    window = closes[-periods - 1 :]
    for previous, current in zip(window[:-1], window[1:]):
        if previous > 0 and current > 0:
            returns.append(math.log(current / previous))
    return pstdev(returns) if len(returns) > 1 else 0.0


def _wick_pressure(candles: list[Candle], periods: int = 3) -> float:
    recent = candles[-periods:]
    if not recent:
        return 0.0
    pressures = []
    for candle in recent:
        high = float(candle.high)
        low = float(candle.low)
        open_ = float(candle.open)
        close = float(candle.close)
        width = high - low
        pressures.append((close - open_) / width if width > 0 else 0.0)
    return sum(pressures) / len(pressures)


def _depth(levels) -> float:
    return float(levels[0].quantity) if levels else 0.0


def _market_probability(yes_ask: Optional[Decimal], no_ask: Optional[Decimal]) -> float:
    if yes_ask is None or no_ask is None:
        return 0.5
    total = yes_ask + no_ask
    if total <= 0:
        return 0.5
    return float(yes_ask / total)


def build_feature_snapshot(
    *,
    candles: list[Candle],
    orderbook: OrderBookSnapshot,
    contract: ContractRef,
    now: datetime,
    raw_fusion_probability: float,
    spread_at_open: Optional[float] = None,
) -> KalshiFeatureSnapshot:
    closes = [float(candle.close) for candle in candles]
    spot_price = closes[-1] if closes else 0.0
    yes_bid = orderbook.best_yes_bid or contract.yes_bid
    yes_ask = orderbook.best_yes_ask or contract.yes_ask
    no_bid = orderbook.best_no_bid or contract.no_bid
    no_ask = orderbook.best_no_ask or contract.no_ask

    yes_spread = float((yes_ask - yes_bid) * 100) if yes_bid is not None and yes_ask is not None else 99.0
    no_spread = float((no_ask - no_bid) * 100) if no_bid is not None and no_ask is not None else 99.0
    spread_cents = min(yes_spread, no_spread)
    market_probability = _market_probability(yes_ask, no_ask)

    close_time = contract.close_time
    seconds_to_expiry = 0.0
    if close_time:
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        seconds_to_expiry = max(0.0, (close_time - now).total_seconds())

    vol_3m = _realized_vol(closes, 3)
    vol_5m = _realized_vol(closes, 5)
    vol_15m = _realized_vol(closes, 15)
    vol_acceleration = vol_3m / vol_15m if vol_15m > 0 else 0.0

    market_midpoint = float(orderbook.yes_midpoint) if orderbook.yes_midpoint is not None else None
    spread_change_pct = (spread_cents - spread_at_open) / spread_at_open if spread_at_open and spread_at_open > 0 else 0.0
    momentum_decay = _return_over(closes, 1) - _return_over(closes, 5)

    features = {
        "spot_price": spot_price,
        "spot_return_30s": ((float(candles[-1].close) / float(candles[-1].open)) - 1.0) if candles and candles[-1].open > 0 else 0.0,
        "spot_return_1m": _return_over(closes, 1),
        "spot_return_3m": _return_over(closes, 3),
        "spot_return_5m": _return_over(closes, 5),
        "realized_vol_3m": vol_3m,
        "realized_vol_5m": vol_5m,
        "realized_vol_15m": vol_15m,
        "volatility_acceleration": vol_acceleration,
        "wick_pressure_3m": _wick_pressure(candles, 3),
        "minutes_to_expiry": seconds_to_expiry / 60.0,
        "seconds_to_expiry": seconds_to_expiry,
        "yes_bid": _safe_float(yes_bid) or 0.0,
        "yes_ask": _safe_float(yes_ask) or 0.0,
        "no_bid": _safe_float(no_bid) or 0.0,
        "no_ask": _safe_float(no_ask) or 0.0,
        "spread_cents": spread_cents,
        "top_yes_depth": _depth(orderbook.yes_bids),
        "top_no_depth": _depth(orderbook.no_bids),
        "top_depth": min(_depth(orderbook.yes_bids), _depth(orderbook.no_bids)),
        "book_imbalance": orderbook_imbalance(orderbook),
        "market_probability": market_probability,
        "market_midpoint": market_midpoint if market_midpoint is not None else 0.5,
        "raw_fusion_probability": raw_fusion_probability,
        "market_probability_gap": raw_fusion_probability - market_probability,
        "momentum_decay": momentum_decay,
        "spread_change_pct": spread_change_pct,
    }

    return KalshiFeatureSnapshot(
        features=features,
        metadata={
            "contract": contract.ticker,
            "contract_open_time": contract.open_time.isoformat() if contract.open_time else None,
            "contract_close_time": contract.close_time.isoformat() if contract.close_time else None,
        },
    )

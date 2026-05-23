"""
Technical indicators and Kalshi-specific market signals for crypto 15m markets.
"""
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, List, Optional

from execution.kalshi_api import OrderBookSnapshot


@dataclass
class Candle:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")


@dataclass
class IndicatorSnapshot:
    rsi: Optional[float]
    macd_histogram: Optional[float]
    cci: Optional[float]
    fisher: Optional[float]
    adx: Optional[float]
    atr: Optional[float]


@dataclass
class SignalValue:
    name: str
    value: float
    confidence: float
    metadata: Dict[str, float]


@dataclass
class DeribitSignalInput:
    """A snapshot of Deribit data to convert into signals."""
    funding_rate_pct: float
    funding_rate_direction: int       # -1, 0, or +1
    funding_confidence: float         # 0-1
    oi_direction: int                # -1, 0, or +1
    oi_change_1h_pct: float
    oi_confidence: float             # 0-1


@dataclass
class SentimentSignalInput:
    score: float
    classification: str = ""


def deribit_signals(d: DeribitSignalInput) -> List[SignalValue]:
    """
    Convert Deribit funding rate and OI data into SignalValue objects.

    - funding_rate_direction: positive → institutional funding long (bullish),
      negative → funding short (bearish).  Funding > 0.01% or < -0.01% is significant.
    - oi_spike: OI increasing → smart money accumulating; OI decreasing → unwinding.
    - funding_oi_divergence: detects mismatch between funding direction and OI
      direction (smart money rotating vs retail positioning).

    Example:
        snapshot = await deribit.get_snapshot()
        signals = deribit_signals(DeribitSignalInput(
            funding_rate_pct=snapshot.funding_rate_pct,
            funding_rate_direction=snapshot.funding_rate_direction,
            funding_confidence=snapshot.funding_confidence,
            oi_direction=snapshot.oi_direction,
            oi_change_1h_pct=snapshot.oi_change_1h_pct,
            oi_confidence=snapshot.oi_confidence,
        ))
    """
    signals: List[SignalValue] = []

    # 1. Funding rate direction signal
    if d.funding_confidence > 0:
        signals.append(SignalValue(
            "funding_rate",
            float(d.funding_rate_direction),
            d.funding_confidence,
            {"funding_rate_pct": d.funding_rate_pct, "confidence": d.funding_confidence},
        ))

    # 2. Open interest spike signal
    if d.oi_confidence > 0:
        signals.append(SignalValue(
            "open_interest",
            float(d.oi_direction),
            d.oi_confidence,
            {"oi_change_1h_pct": d.oi_change_1h_pct, "confidence": d.oi_confidence},
        ))

    # 3. Funding/OI divergence signal
    # When funding and OI point in opposite directions, it's a strong signal
    # of institutional vs retail disagreement.
    if d.funding_rate_direction and d.oi_direction and d.funding_confidence > 0 and d.oi_confidence > 0:
        divergence = d.funding_rate_direction * d.oi_direction
        divergence_value = float(d.funding_rate_direction if divergence > 0 else -d.funding_rate_direction)
        signals.append(SignalValue(
            "funding_oi_divergence",
            divergence_value,
            min(d.funding_confidence, d.oi_confidence),
            {"funding_dir": d.funding_rate_direction, "oi_dir": d.oi_direction, "divergent": 1.0 if divergence < 0 else 0.0},
        ))

    return signals


def sentiment_signals(sentiment: SentimentSignalInput) -> List[SignalValue]:
    score = max(0.0, min(100.0, sentiment.score))
    if score <= 25:
        value = 1.0
    elif score >= 75:
        value = -1.0
    else:
        value = (50.0 - score) / 25.0
    confidence = min(1.0, abs(score - 50.0) / 50.0)
    if confidence <= 0:
        return []
    return [
        SignalValue(
            "fear_greed",
            max(-1.0, min(1.0, value)),
            confidence,
            {"sentiment_score": score},
        )
    ]


def _floats(values: Iterable[Decimal]) -> List[float]:
    return [float(value) for value in values]


def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    alpha = 2.0 / (period + 1)
    current = sum(values[:period]) / period
    for value in values[period:]:
        current = (value * alpha) + (current * (1 - alpha))
    return current


def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) <= period:
        return None
    gains = []
    losses = []
    for prev, current in zip(closes[-period - 1 : -1], closes[-period:]):
        delta = current - prev
        gains.append(max(delta, 0.0))
        losses.append(abs(min(delta, 0.0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd_histogram(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[float]:
    if len(closes) < slow + signal:
        return None
    macd_values = []
    for idx in range(slow, len(closes) + 1):
        window = closes[:idx]
        fast_ema = ema(window, fast)
        slow_ema = ema(window, slow)
        if fast_ema is not None and slow_ema is not None:
            macd_values.append(fast_ema - slow_ema)
    if len(macd_values) < signal:
        return None
    signal_line = ema(macd_values, signal)
    if signal_line is None:
        return None
    return macd_values[-1] - signal_line


def cci(candles: List[Candle], period: int = 20) -> Optional[float]:
    if len(candles) < period:
        return None
    typical = [(float(c.high + c.low + c.close) / 3.0) for c in candles[-period:]]
    sma = sum(typical) / period
    mean_deviation = sum(abs(value - sma) for value in typical) / period
    if mean_deviation == 0:
        return 0.0
    return (typical[-1] - sma) / (0.015 * mean_deviation)


def fisher_transform(candles: List[Candle], period: int = 10) -> Optional[float]:
    if len(candles) < period:
        return None
    highs = [float(c.high) for c in candles[-period:]]
    lows = [float(c.low) for c in candles[-period:]]
    close = float(candles[-1].close)
    high = max(highs)
    low = min(lows)
    if high == low:
        return 0.0
    value = 2.0 * ((close - low) / (high - low) - 0.5)
    value = max(-0.999, min(0.999, value))
    return 0.5 * math.log((1 + value) / (1 - value))


def true_ranges(candles: List[Candle]) -> List[float]:
    ranges = []
    for idx in range(1, len(candles)):
        high = float(candles[idx].high)
        low = float(candles[idx].low)
        prev_close = float(candles[idx - 1].close)
        ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return ranges


def atr(candles: List[Candle], period: int = 14) -> Optional[float]:
    ranges = true_ranges(candles)
    if len(ranges) < period:
        return None
    return sum(ranges[-period:]) / period


def adx(candles: List[Candle], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    plus_dm = []
    minus_dm = []
    tr = []
    for idx in range(1, len(candles)):
        high = float(candles[idx].high)
        low = float(candles[idx].low)
        prev_high = float(candles[idx - 1].high)
        prev_low = float(candles[idx - 1].low)
        prev_close = float(candles[idx - 1].close)

        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

    recent_tr = sum(tr[-period:])
    if recent_tr == 0:
        return 0.0
    plus_di = 100.0 * (sum(plus_dm[-period:]) / recent_tr)
    minus_di = 100.0 * (sum(minus_dm[-period:]) / recent_tr)
    denom = plus_di + minus_di
    if denom == 0:
        return 0.0
    return 100.0 * abs(plus_di - minus_di) / denom


def timeframe_disagreement_signal(candles_1m: List[Candle]) -> Optional[SignalValue]:
    """
    Compare 1m RSI direction vs 5m RSI direction.
    Returns a signal when they disagree, with magnitude proportional to disagreement strength.
    """
    if len(candles_1m) < 35:
        return None

    closes_1m = _floats(c.close for c in candles_1m)
    rsi_1m = rsi(closes_1m, period=14)
    if rsi_1m is None:
        return None

    if len(candles_1m) < 60:
        return None

    candles_5m = candles_1m[::5]
    if len(candles_5m) < 10:
        return None

    closes_5m = _floats(c.close for c in candles_5m)
    rsi_5m = rsi(closes_5m, period=14)
    if rsi_5m is None:
        return None

    dir_1m = 1.0 if rsi_1m >= 50.0 else -1.0
    dir_5m = 1.0 if rsi_5m >= 50.0 else -1.0

    if dir_1m == dir_5m:
        return None

    strength_1m = abs(rsi_1m - 50.0) / 30.0
    strength_5m = abs(rsi_5m - 50.0) / 30.0
    magnitude = min(1.0, (strength_1m + strength_5m) / 2.0)
    disagreement_value = -dir_1m * magnitude

    return SignalValue(
        "timeframe_disagreement",
        max(-1.0, min(1.0, disagreement_value)),
        magnitude,
        {"rsi_1m": rsi_1m, "rsi_5m": rsi_5m},
    )


def compute_indicators(candles: List[Candle]) -> IndicatorSnapshot:
    closes = _floats(c.close for c in candles)
    return IndicatorSnapshot(
        rsi=rsi(closes),
        macd_histogram=macd_histogram(closes),
        cci=cci(candles),
        fisher=fisher_transform(candles),
        adx=adx(candles),
        atr=atr(candles),
    )


def normalize_indicators(snapshot: IndicatorSnapshot, candles: List[Candle]) -> List[SignalValue]:
    signals: List[SignalValue] = []
    close = float(candles[-1].close) if candles else 0.0

    if snapshot.rsi is not None:
        value = max(-1.0, min(1.0, (50.0 - snapshot.rsi) / 30.0))
        signals.append(SignalValue("rsi", value, min(1.0, abs(value)), {"rsi": snapshot.rsi}))

    if snapshot.macd_histogram is not None and close > 0:
        value = max(-1.0, min(1.0, snapshot.macd_histogram / (close * 0.001)))
        signals.append(SignalValue("macd", value, min(1.0, abs(value)), {"macd_histogram": snapshot.macd_histogram}))

    if snapshot.cci is not None:
        value = max(-1.0, min(1.0, snapshot.cci / 200.0))
        signals.append(SignalValue("cci", value, min(1.0, abs(value)), {"cci": snapshot.cci}))

    if snapshot.fisher is not None:
        value = max(-1.0, min(1.0, snapshot.fisher / 3.0))
        signals.append(SignalValue("fisher", value, min(1.0, abs(value)), {"fisher": snapshot.fisher}))

    if snapshot.adx is not None and snapshot.macd_histogram is not None:
        trend_direction = 1.0 if snapshot.macd_histogram >= 0 else -1.0
        value = trend_direction * max(0.0, min(1.0, (snapshot.adx - 20.0) / 30.0))
        signals.append(SignalValue("adx", value, min(1.0, abs(value)), {"adx": snapshot.adx}))

    if snapshot.atr is not None and close > 0:
        atr_pct = snapshot.atr / close
        value = -max(0.0, min(1.0, (atr_pct - 0.0015) / 0.004))
        signals.append(SignalValue("atr_regime", value, min(1.0, abs(value)), {"atr": snapshot.atr, "atr_pct": atr_pct}))

    return signals


def orderbook_imbalance(orderbook: OrderBookSnapshot, top_levels: int = 10) -> float:
    yes_bid_volume = sum(level.price * level.quantity for level in orderbook.yes_bids[:top_levels])
    yes_ask_volume = sum((Decimal("1") - level.price) * level.quantity for level in orderbook.no_bids[:top_levels])
    total = yes_bid_volume + yes_ask_volume
    if total <= 0:
        return 0.0
    return float((yes_bid_volume - yes_ask_volume) / total)


def orderbook_imbalance_multi_depth(book: OrderBookSnapshot) -> List[SignalValue]:
    signals = []
    for depth in [3, 5, 10, 20]:
        yes_vol = sum(level.price * level.quantity for level in book.yes_bids[:depth])
        no_vol = sum((Decimal("1") - level.price) * level.quantity for level in book.no_bids[:depth])
        total = yes_vol + no_vol
        imbalance = float((yes_vol - no_vol) / total) if total > 0 else 0.0
        signals.append(SignalValue(
            f"book_imbalance_d{depth}",
            max(-1.0, min(1.0, imbalance)),
            min(1.0, abs(imbalance)),
            {"imbalance": imbalance},
        ))
    return signals


def implied_market_probability(yes_price: Decimal, no_price: Decimal) -> float:
    total = yes_price + no_price
    if total <= 0:
        return 0.5
    return float(yes_price / total)


def kalshi_market_signals(orderbook: OrderBookSnapshot) -> List[SignalValue]:
    signals = []
    signals.extend(orderbook_imbalance_multi_depth(orderbook))

    yes_ask = orderbook.best_yes_ask
    no_ask = orderbook.best_no_ask
    if yes_ask is not None and no_ask is not None:
        market_prob = implied_market_probability(yes_ask, no_ask)
        distance_from_50 = abs(market_prob - 0.5)
        crowd_direction = 1.0 if market_prob < 0.5 else -1.0
        crowd_fade_magnitude = min(1.0, distance_from_50 * 3)
        crowd_value = crowd_direction * crowd_fade_magnitude
        signals.append(
            SignalValue(
                "market_probability",
                max(-1.0, min(1.0, (market_prob - 0.5) * 2.0)),
                abs(market_prob - 0.5) * 2.0,
                {"market_probability": market_prob},
            )
        )
        signals.append(
            SignalValue(
                "crowd_fade",
                crowd_value,
                abs(crowd_value),
                {"market_probability": market_prob},
            )
        )

    return signals

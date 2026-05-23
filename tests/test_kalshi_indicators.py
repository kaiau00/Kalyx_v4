from datetime import datetime, timezone
from decimal import Decimal

from core.strategy_brain.kalshi_indicators import (
    Candle,
    DeribitSignalInput,
    SentimentSignalInput,
    compute_indicators,
    deribit_signals,
    implied_market_probability,
    kalshi_market_signals,
    orderbook_imbalance,
    orderbook_imbalance_multi_depth,
    sentiment_signals,
)
from execution.kalshi_api import OrderBookLevel, OrderBookSnapshot


def make_candles(count=60):
    candles = []
    price = Decimal("100")
    for idx in range(count):
        price += Decimal("0.25")
        candles.append(
            Candle(
                open=price - Decimal("0.10"),
                high=price + Decimal("0.50"),
                low=price - Decimal("0.50"),
                close=price,
                volume=Decimal("100"),
            )
        )
    return candles


def test_compute_indicators_returns_core_values():
    snapshot = compute_indicators(make_candles())

    assert snapshot.rsi is not None
    assert snapshot.macd_histogram is not None
    assert snapshot.cci is not None
    assert snapshot.fisher is not None
    assert snapshot.adx is not None
    assert snapshot.atr is not None


def test_orderbook_imbalance_uses_yes_bids_and_synthetic_yes_asks():
    book = OrderBookSnapshot(
        ticker="TEST",
        yes_bids=[OrderBookLevel(Decimal("0.60"), Decimal("100"))],
        no_bids=[OrderBookLevel(Decimal("0.30"), Decimal("200"))],
        timestamp=datetime.now(timezone.utc),
    )

    imbalance = orderbook_imbalance(book)

    assert imbalance < 0


def test_orderbook_imbalance_multi_depth():
    book = OrderBookSnapshot(
        ticker="TEST",
        yes_bids=[
            OrderBookLevel(Decimal("0.60"), Decimal("100")),
            OrderBookLevel(Decimal("0.58"), Decimal("80")),
            OrderBookLevel(Decimal("0.55"), Decimal("60")),
            OrderBookLevel(Decimal("0.52"), Decimal("40")),
            OrderBookLevel(Decimal("0.50"), Decimal("20")),
        ],
        no_bids=[
            OrderBookLevel(Decimal("0.30"), Decimal("200")),
            OrderBookLevel(Decimal("0.32"), Decimal("150")),
            OrderBookLevel(Decimal("0.35"), Decimal("100")),
            OrderBookLevel(Decimal("0.38"), Decimal("50")),
            OrderBookLevel(Decimal("0.40"), Decimal("10")),
        ],
        timestamp=datetime.now(timezone.utc),
    )

    signals = orderbook_imbalance_multi_depth(book)
    signal_names = [s.name for s in signals]

    assert "book_imbalance_d3" in signal_names
    assert "book_imbalance_d5" in signal_names
    assert "book_imbalance_d10" in signal_names
    assert "book_imbalance_d20" in signal_names

    d3_signal = next(s for s in signals if s.name == "book_imbalance_d3")
    assert d3_signal.value < 0


def test_implied_market_probability_from_yes_and_no_prices():
    assert implied_market_probability(Decimal("0.55"), Decimal("0.45")) == 0.55


def test_kalshi_market_signals_include_crowd_fade():
    book = OrderBookSnapshot(
        ticker="TEST",
        yes_bids=[OrderBookLevel(Decimal("0.65"), Decimal("10"))],
        no_bids=[OrderBookLevel(Decimal("0.30"), Decimal("10"))],
        timestamp=datetime.now(timezone.utc),
    )

    signals = {signal.name: signal for signal in kalshi_market_signals(book)}

    assert "orderbook_imbalance" not in signals
    assert "book_imbalance_d3" in signals
    assert "market_probability" in signals
    assert "crowd_fade" in signals
    assert signals["crowd_fade"].value < 0


def test_crowd_fade_continuous_at_62_percent():
    book = OrderBookSnapshot(
        ticker="TEST",
        yes_bids=[OrderBookLevel(Decimal("0.38"), Decimal("10"))],
        no_bids=[OrderBookLevel(Decimal("0.62"), Decimal("10"))],
        timestamp=datetime.now(timezone.utc),
    )

    signals = {signal.name: signal for signal in kalshi_market_signals(book)}
    crowd_fade = signals["crowd_fade"]

    assert crowd_fade.value > 0
    assert 0.0 < crowd_fade.value < 0.5


def test_crowd_fade_strong_at_90_percent():
    book = OrderBookSnapshot(
        ticker="TEST",
        yes_bids=[OrderBookLevel(Decimal("0.10"), Decimal("10"))],
        no_bids=[OrderBookLevel(Decimal("0.90"), Decimal("10"))],
        timestamp=datetime.now(timezone.utc),
    )

    signals = {signal.name: signal for signal in kalshi_market_signals(book)}
    crowd_fade = signals["crowd_fade"]

    assert crowd_fade.value > 0
    assert crowd_fade.value > 0.7


def test_crowd_fade_positive_when_below_50():
    book = OrderBookSnapshot(
        ticker="TEST",
        yes_bids=[OrderBookLevel(Decimal("0.70"), Decimal("10"))],
        no_bids=[OrderBookLevel(Decimal("0.30"), Decimal("10"))],
        timestamp=datetime.now(timezone.utc),
    )

    signals = {signal.name: signal for signal in kalshi_market_signals(book)}
    crowd_fade = signals["crowd_fade"]

    assert crowd_fade.value < 0


def test_deribit_signals_include_divergence():
    signals = {signal.name: signal for signal in deribit_signals(
        DeribitSignalInput(
            funding_rate_pct=0.02,
            funding_rate_direction=1,
            funding_confidence=0.4,
            oi_direction=-1,
            oi_change_1h_pct=-0.05,
            oi_confidence=0.5,
        )
    )}

    assert "funding_rate" in signals
    assert "open_interest" in signals
    assert "funding_oi_divergence" in signals
    assert signals["funding_oi_divergence"].value == -1.0


def test_sentiment_signal_fades_greed():
    signals = sentiment_signals(SentimentSignalInput(score=80))

    assert len(signals) == 1
    assert signals[0].name == "fear_greed"
    assert signals[0].value < 0

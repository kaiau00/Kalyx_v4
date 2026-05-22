from datetime import datetime, timezone
from decimal import Decimal

from core.strategy_brain.kalshi_features import build_feature_snapshot, _return_over, _realized_vol, _wick_pressure
from core.strategy_brain.kalshi_indicators import Candle
from execution.kalshi_api import ContractRef, OrderBookLevel, OrderBookSnapshot


def make_candles(count=60, start_price=100.0):
    candles = []
    price = Decimal(str(start_price))
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


def test_momentum_decay_feature_in_snapshot():
    candles = make_candles(60, start_price=100.0)
    book = OrderBookSnapshot(
        ticker="TEST",
        yes_bids=[OrderBookLevel(Decimal("0.60"), Decimal("100"))],
        no_bids=[OrderBookLevel(Decimal("0.40"), Decimal("100"))],
        timestamp=datetime.now(timezone.utc),
    )
    contract = ContractRef(
        ticker="TEST",
        event_ticker=None,
        title="Test",
        open_time=datetime.now(timezone.utc),
        close_time=datetime.now(timezone.utc),
        status="open",
    )
    now = datetime.now(timezone.utc)
    snapshot = build_feature_snapshot(
        candles=candles,
        orderbook=book,
        contract=contract,
        now=now,
        raw_fusion_probability=0.55,
        spread_at_open=5.0,
    )

    assert "momentum_decay" in snapshot.features
    assert "spread_change_pct" in snapshot.features
    assert "market_midpoint" in snapshot.features


def test_spread_change_pct_feature_exists():
    candles = make_candles(60, start_price=100.0)
    book = OrderBookSnapshot(
        ticker="TEST",
        yes_bids=[OrderBookLevel(Decimal("0.60"), Decimal("100"))],
        no_bids=[OrderBookLevel(Decimal("0.40"), Decimal("100"))],
        timestamp=datetime.now(timezone.utc),
    )
    contract = ContractRef(
        ticker="TEST",
        event_ticker=None,
        title="Test",
        open_time=datetime.now(timezone.utc),
        close_time=datetime.now(timezone.utc),
        status="open",
    )
    now = datetime.now(timezone.utc)
    snapshot = build_feature_snapshot(
        candles=candles,
        orderbook=book,
        contract=contract,
        now=now,
        raw_fusion_probability=0.55,
        spread_at_open=5.0,
    )

    assert "spread_change_pct" in snapshot.features
    assert "momentum_decay" in snapshot.features


def test_market_midpoint_present_when_orderbook_provided():
    book = OrderBookSnapshot(
        ticker="TEST",
        yes_bids=[OrderBookLevel(Decimal("0.55"), Decimal("100"))],
        no_bids=[OrderBookLevel(Decimal("0.45"), Decimal("100"))],
        timestamp=datetime.now(timezone.utc),
    )

    midpoint = book.yes_midpoint
    assert midpoint is not None
    assert 0.0 < float(midpoint) < 1.0
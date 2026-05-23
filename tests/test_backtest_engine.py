from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from core.backtesting.backtest_engine import (
    BacktestConfig,
    MarketSimulator,
    PositionTracker,
    _temporary_strategy_env,
)
from core.backtesting.historical_data import HistoricalDataLoader


def test_position_tracker_resolves_yes_contract_with_binary_settlement():
    tracker = PositionTracker()
    trade = tracker.open_position(
        window_id="w1",
        ticker="SIM_W1",
        side="yes",
        quantity=2,
        entry_price=Decimal("0.55"),
        fee_per_contract=Decimal("0.10"),
        slippage_cents=0.5,
        market_probability=0.52,
        predicted_probability=0.60,
        model_confidence=0.30,
        intent_edge=Decimal("0.05"),
        signals=[{"name": "rsi", "value": 0.4}],
        ground_truth="yes",
        timestamp=datetime.now(timezone.utc),
    )

    pnl = tracker.resolve_position("SIM_W1", "yes", datetime.now(timezone.utc))

    assert pnl == Decimal("0.70")
    assert trade.won is True
    assert trade.settlement_pnl == Decimal("0.70")
    assert tracker.total_pnl == Decimal("0.70")
    assert tracker.win_rate == 1.0


def test_position_tracker_resolves_no_contract_with_binary_settlement():
    tracker = PositionTracker()
    trade = tracker.open_position(
        window_id="w2",
        ticker="SIM_W2",
        side="no",
        quantity=1,
        entry_price=Decimal("0.40"),
        fee_per_contract=Decimal("0.10"),
        slippage_cents=0.5,
        market_probability=0.48,
        predicted_probability=0.42,
        model_confidence=0.25,
        intent_edge=Decimal("0.03"),
        signals=[{"name": "crowd_fade", "value": -0.5}],
        ground_truth="no",
        timestamp=datetime.now(timezone.utc),
    )

    pnl = tracker.resolve_position("SIM_W2", "no", datetime.now(timezone.utc))

    assert pnl == Decimal("0.50")
    assert trade.won is True
    assert trade.settlement_pnl == Decimal("0.50")
    assert tracker.total_pnl == Decimal("0.50")


def test_market_simulator_honors_requested_market_probability():
    simulator = MarketSimulator(BacktestConfig(synthetic_spread_cents=4.0), regime="bull_2024")

    class Window:
        open_time = datetime.now(timezone.utc)

    book = simulator.simulate_orderbook("SIM_X", Window(), market_probability=0.61)

    assert book.yes_midpoint == Decimal("0.61")
    assert book.best_yes_bid < Decimal("0.61")
    assert book.best_yes_ask > Decimal("0.61")


@pytest.mark.asyncio
async def test_historical_loader_uses_explicit_regime_for_synthetic_fallback():
    loader = HistoricalDataLoader(regime="bear")
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    candles = await loader.load_candles(start, end)
    await loader.close()

    assert candles
    assert loader.last_data_source == "synthetic"
    assert loader.last_synthetic_used is True


def test_temporary_strategy_env_uses_backtest_config_values(tmp_path):
    config = BacktestConfig(
        bankroll=2500.0,
        max_trade_fraction=0.15,
        kelly_multiplier=0.35,
        max_trade_dollars=40.0,
        max_daily_loss=80.0,
        max_total_exposure=120.0,
        ev_threshold=0.02,
        fee_per_contract=0.01,
        min_edge_after_fees=0.005,
        candle_limit=90,
    )

    with _temporary_strategy_env(tmp_path, config) as env:
        assert env["KALSHI_BANKROLL"] == "2500.0"
        assert env["KALSHI_KELLY_CAP"] == "0.15"
        assert env["KALSHI_KELLY_MULTIPLIER"] == "0.35"
        assert env["KALSHI_MAX_TRADE_DOLLARS"] == "40.0"
        assert env["KALSHI_MAX_DAILY_LOSS"] == "80.0"
        assert env["KALSHI_MAX_TOTAL_EXPOSURE"] == "120.0"
        assert env["KALSHI_EV_THRESHOLD"] == "0.02"
        assert env["KALSHI_FEE_PER_CONTRACT"] == "0.01"
        assert env["KALSHI_MIN_EDGE_AFTER_FEES"] == "0.005"
        assert env["KALSHI_CANDLE_LIMIT"] == "90"

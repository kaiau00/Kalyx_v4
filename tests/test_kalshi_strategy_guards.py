import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal

from core.strategy_brain.kalshi_fusion import KalshiSignalFusion
from core.strategy_brain.kalshi_model import KalshiModelPrediction
from core.strategy_brain.strategies.kalshi_multisignal_strategy import (
    _build_settlement_update,
    load_risk_config,
    run_kalshi_multisignal_strategy,
)
from execution.kalshi_api import ContractRef, OrderBookLevel, OrderBookSnapshot


class FakeCandleSource:
    async def get_klines(self, interval: str, limit: int):
        return [
            {
                "open": Decimal("100"),
                "high": Decimal("101"),
                "low": Decimal("99"),
                "close": Decimal("100"),
                "volume": Decimal("1"),
            }
            for _ in range(limit)
        ]

    async def close(self):
        return None


class FakeKalshi:
    def __init__(self):
        self.placed_orders = []

    async def get_active_15m_contract(self, asset: str = "BTC", now=None):
        return ContractRef(
            ticker="BTC-TEST",
            event_ticker="BTC-TEST",
            title="Bitcoin 15m",
            open_time=datetime(2026, 5, 22, 6, 0, tzinfo=timezone.utc),
            close_time=datetime(2026, 5, 22, 6, 15, tzinfo=timezone.utc),
            status="open",
            yes_ask=Decimal("0.45"),
            no_ask=Decimal("0.55"),
        )

    async def get_orderbook(self, contract_code: str, depth: int = 20):
        return OrderBookSnapshot(
            ticker=contract_code,
            yes_bids=[OrderBookLevel(price=Decimal("0.44"), quantity=Decimal("10"))],
            no_bids=[OrderBookLevel(price=Decimal("0.54"), quantity=Decimal("10"))],
            timestamp=datetime.now(timezone.utc),
        )

    async def get_positions(self):
        return []

    async def get_settlements(self, start=None):
        return []

    async def close(self):
        return None


class FakeModel:
    def predict(self, snapshot, fallback_prob: float, fallback_confidence: float):
        return KalshiModelPrediction(
            predicted_prob=0.80,
            confidence=0.90,
            model_version="fake",
            fallback_used=False,
            reason="ok",
        )


def test_strategy_skips_contract_that_was_already_traded(tmp_path, monkeypatch):
    trade_log = tmp_path / "trades.jsonl"
    trade_log.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-22T06:05:00+00:00",
                "contract": "BTC-TEST",
                "dry_run": False,
                "trade_intent": {"should_trade": True, "side": "yes", "quantity": 1, "limit_price": "0.45"},
                "order": {"order_id": "ord_1", "status": "resting"},
            }
        )
        + "\n"
    )
    monkeypatch.setenv("KALSHI_TRADE_LOG", str(trade_log))

    result = asyncio.run(
        run_kalshi_multisignal_strategy(
            kalshi=FakeKalshi(),
            candle_source=FakeCandleSource(),
            fusion=KalshiSignalFusion(signal_names=["edge"], state_path=str(tmp_path / "state.json")),
            dry_run=False,
        )
    )

    assert result.reason == "already_traded_contract"


def test_settlement_update_ignores_unfilled_live_order():
    record = {
        "contract": "BTC-TEST",
        "trade_intent": {"should_trade": True, "quantity": 3, "side": "yes", "limit_price": "0.42"},
        "order": {"order_id": "ord_123", "status": "resting"},
    }

    assert _build_settlement_update(record, "yes") is None


def test_settlement_update_uses_filled_order_quantity():
    record = {
        "contract": "BTC-TEST",
        "trade_intent": {"should_trade": True, "quantity": 3, "side": "yes", "limit_price": "0.42"},
        "order": {"order_id": "ord_123", "status": "filled", "filled_quantity": "2"},
    }

    update = _build_settlement_update(record, "yes")

    assert update is not None
    assert update.quantity == 2


def test_strategy_logs_feature_snapshot_and_model_prediction(tmp_path, monkeypatch):
    trade_log = tmp_path / "trades.jsonl"
    monkeypatch.setenv("KALSHI_TRADE_LOG", str(trade_log))
    monkeypatch.setenv("KALSHI_FEE_PER_CONTRACT", "0.01")
    monkeypatch.setenv("KALSHI_MIN_EDGE_AFTER_FEES", "0.01")

    result = asyncio.run(
        run_kalshi_multisignal_strategy(
            kalshi=FakeKalshi(),
            candle_source=FakeCandleSource(),
            fusion=KalshiSignalFusion(state_path=str(tmp_path / "state.json")),
            probability_model=FakeModel(),
            dry_run=True,
        )
    )
    record = json.loads(trade_log.read_text().splitlines()[0])

    assert result.predicted_prob == 0.80
    assert record["model_version"] == "fake"
    assert "minutes_to_expiry" in record["features"]
    assert "edge_after_fees" in record


def test_load_risk_config_reads_live_entry_guard_env(monkeypatch):
    monkeypatch.setenv("KALSHI_EV_THRESHOLD", "0.02")
    monkeypatch.setenv("KALSHI_MIN_EDGE_AFTER_FEES", "0.05")
    monkeypatch.setenv("KALSHI_MIN_MODEL_CONFIDENCE", "0.50")
    monkeypatch.setenv("KALSHI_HARD_LATE_ENTRY_CUTOFF_SECONDS", "180")
    monkeypatch.setenv("KALSHI_MIN_PROBABILITY_GAP", "0.05")
    monkeypatch.setenv("KALSHI_MIN_MARKET_SIDE_PROB", "0.08")
    monkeypatch.setenv("KALSHI_MAX_MARKET_SIDE_PROB", "0.92")
    monkeypatch.setenv("KALSHI_LATE_WINDOW_SECONDS", "420")
    monkeypatch.setenv("KALSHI_LATE_WINDOW_MIN_EDGE", "0.08")
    monkeypatch.setenv("KALSHI_LATE_WINDOW_MIN_CONFIDENCE", "0.60")

    config = load_risk_config()

    assert config.ev_threshold == Decimal("0.02")
    assert config.min_edge_after_fees == Decimal("0.05")
    assert config.min_model_confidence == Decimal("0.50")
    assert config.hard_late_entry_cutoff_seconds == 180
    assert config.min_probability_gap == Decimal("0.05")
    assert config.min_market_side_prob == Decimal("0.08")
    assert config.max_market_side_prob == Decimal("0.92")
    assert config.late_window_seconds == 420
    assert config.late_window_min_edge == Decimal("0.08")
    assert config.late_window_min_confidence == Decimal("0.60")

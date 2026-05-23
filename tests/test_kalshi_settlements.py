import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from core.strategy_brain.strategies.kalshi_multisignal_strategy import (
    SettlementUpdate,
    update_kalshi_weights_from_settlements,
)
from execution.kalshi_api import SettlementResult


class FakeKalshi:
    def __init__(self, settlements):
        self._settlements = settlements

    async def get_settlements(self):
        return self._settlements

    async def close(self):
        return None


def test_settlement_reconciliation_prefers_live_trade_record(tmp_path, monkeypatch):
    trade_log = tmp_path / "trades.jsonl"
    learned = tmp_path / "learned.json"
    state = tmp_path / "state.json"

    first = {
        "timestamp": "2026-05-22T06:00:00+00:00",
        "contract": "BTC-TEST",
        "signals": [{"name": "rsi", "value": 0.5, "confidence": 1.0, "metadata": {}}],
        "trade_intent": {"should_trade": False, "quantity": 0, "side": None, "limit_price": "0"},
        "order": None,
    }
    second = {
        "timestamp": "2026-05-22T06:10:00+00:00",
        "contract": "BTC-TEST",
        "signals": [{"name": "rsi", "value": 0.8, "confidence": 1.0, "metadata": {}}],
        "trade_intent": {"should_trade": True, "quantity": 3, "side": "yes", "limit_price": "0.42"},
        "order": {"order_id": "ord_123", "status": "filled", "side": "yes"},
    }
    trade_log.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    monkeypatch.setenv("KALSHI_TRADE_LOG", str(trade_log))
    monkeypatch.setenv("KALSHI_LEARNED_SETTLEMENTS", str(learned))
    monkeypatch.setenv("KALSHI_SIGNAL_STATE", str(state))
    monkeypatch.setenv("KALSHI_FEE_PER_CONTRACT", "0.10")

    settlements = [
        SettlementResult(
            ticker="BTC-TEST",
            result="yes",
            settled_time=datetime.now(timezone.utc),
            raw={},
        )
    ]

    updates = asyncio.run(
        update_kalshi_weights_from_settlements(
            kalshi=FakeKalshi(settlements),
        )
    )

    assert len(updates) == 1
    update = updates[0]
    assert isinstance(update, SettlementUpdate)
    assert update.contract_ticker == "BTC-TEST"
    assert update.side == "yes"
    assert update.quantity == 3
    assert update.won is True

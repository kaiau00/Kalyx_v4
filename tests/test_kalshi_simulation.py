import json

from simulate_kalshi_strategy import run_simulation


def test_simulation_reports_trades_and_pnl(tmp_path):
    rows = []
    for index in range(40):
        base = 100 + index
        rows.append(
            {
                "timestamp": "2026-05-22T06:05:00+00:00",
                "ticker": "BTC-TEST",
                "minutes_to_expiry": 10,
                "candles": [
                    {"open": base + offset, "high": base + offset + 1, "low": base + offset - 1, "close": base + offset + 0.5}
                    for offset in range(40)
                ],
                "orderbook": {"yes": [["0.40", "20"]], "no": [["0.58", "20"]]},
                "result": "yes",
            }
        )
    path = tmp_path / "snapshots.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))

    stats = run_simulation(path, str(tmp_path / "missing-model.json"), slippage_cents=0)

    assert stats["snapshots"] == 40
    assert stats["trades"] > 0
    assert "net_pnl" in stats

#!/usr/bin/env python3
"""
Summarize Kalshi strategy logs by PnL, edge, expiry, spread, and skip reason.
"""
import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable


def _decimal(value: Any, default: str = "0") -> Decimal:
    if value in (None, ""):
        return Decimal(default)
    return Decimal(str(value))


def _bucket(value: Decimal, edges: Iterable[Decimal]) -> str:
    edge_list = list(edges)
    for edge in edge_list:
        if value <= edge:
            return f"<= {edge}"
    return f"> {edge_list[-1]}"


def _record_result(record: Dict[str, Any]) -> str | None:
    result = record.get("result") or record.get("settlement_result")
    return result if result in {"yes", "no"} else None


def _record_pnl(record: Dict[str, Any], fee: Decimal) -> Decimal:
    result = _record_result(record)
    intent = record.get("trade_intent") or {}
    if result not in {"yes", "no"} or not intent.get("should_trade"):
        return Decimal("0")
    side = intent.get("side")
    quantity = int(intent.get("quantity", 0) or 0)
    price = _decimal(intent.get("limit_price"))
    if side not in {"yes", "no"} or quantity <= 0:
        return Decimal("0")
    won = side == result
    per_contract = (Decimal("1") - price - fee) if won else -(price + fee)
    return per_contract * Decimal(quantity)


def analyze(path: Path, fee: Decimal) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "records": 0,
        "trades": 0,
        "skipped": 0,
        "wins": 0,
        "realized_pnl": Decimal("0"),
        "skip_reasons": {},
        "pnl_by_edge_bucket": {},
        "pnl_by_expiry_bucket": {},
        "pnl_by_spread_bucket": {},
        "pnl_by_volatility_bucket": {},
    }

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        summary["records"] += 1
        intent = record.get("trade_intent") or {}
        features = record.get("features") or {}
        if not intent.get("should_trade"):
            summary["skipped"] += 1
            reason = record.get("block_reason") or intent.get("reason") or "unknown"
            summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1
            continue

        summary["trades"] += 1
        result = _record_result(record)
        if result and intent.get("side") == result:
            summary["wins"] += 1
        pnl = _record_pnl(record, fee)
        summary["realized_pnl"] += pnl
        edge_bucket = _bucket(_decimal(record.get("edge_after_fees")), [Decimal("0.03"), Decimal("0.05"), Decimal("0.08"), Decimal("0.12")])
        expiry_bucket = _bucket(_decimal(features.get("minutes_to_expiry")), [Decimal("1"), Decimal("3"), Decimal("5"), Decimal("10")])
        spread_bucket = _bucket(_decimal(features.get("spread_cents")), [Decimal("2"), Decimal("4"), Decimal("6"), Decimal("10")])
        vol_bucket = _bucket(_decimal(features.get("realized_vol_5m")), [Decimal("0.0005"), Decimal("0.001"), Decimal("0.002"), Decimal("0.004")])
        summary["pnl_by_edge_bucket"][edge_bucket] = summary["pnl_by_edge_bucket"].get(edge_bucket, Decimal("0")) + pnl
        summary["pnl_by_expiry_bucket"][expiry_bucket] = summary["pnl_by_expiry_bucket"].get(expiry_bucket, Decimal("0")) + pnl
        summary["pnl_by_spread_bucket"][spread_bucket] = summary["pnl_by_spread_bucket"].get(spread_bucket, Decimal("0")) + pnl
        summary["pnl_by_volatility_bucket"][vol_bucket] = summary["pnl_by_volatility_bucket"].get(vol_bucket, Decimal("0")) + pnl

    summary["win_rate"] = (summary["wins"] / summary["trades"]) if summary["trades"] else 0
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Kalshi trade logs.")
    parser.add_argument("--log", default="kalshi_trades.jsonl", help="Path to strategy JSONL log.")
    parser.add_argument("--fee", default="0.10", help="Fee per contract.")
    args = parser.parse_args()

    print(json.dumps(analyze(Path(args.log), Decimal(str(args.fee))), indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validate ETH and SOL strategy behavior over historical 15m windows.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtesting.historical_data import HistoricalDataLoader
from core.strategy_brain.kalshi_fusion import KalshiSignalFusion
from core.strategy_brain.kalshi_indicators import compute_indicators, normalize_indicators


async def validate_asset(asset: str, start: datetime, end: datetime) -> dict:
    loader = HistoricalDataLoader(
        provider="coinbase",
        coinbase_product={"ETH": "ETH-USD", "SOL": "SOL-USD"}.get(asset.upper(), f"{asset.upper()}-USD"),
        regime="sideways",
    )
    windows = await loader.load_windows(start, end)
    await loader.close()
    fusion = KalshiSignalFusion(state_path=f"/tmp/kalshi_validate_{asset.lower()}.json", min_samples=1)
    correct = 0
    traded = 0
    for window in windows:
        signals = normalize_indicators(compute_indicators(window.candles), window.candles)
        fused = fusion.fuse(signals)
        traded += 1
        correct += int((fused.predicted_prob >= 0.5) == (window.ground_truth == "yes"))
    return {
        "asset": asset.upper(),
        "windows": len(windows),
        "trades_evaluated": traded,
        "directional_accuracy": (correct / traded) if traded else 0.0,
    }


async def main_async(args):
    reports = []
    for asset in args.assets:
        reports.append(await validate_asset(asset, args.start, args.end))
    print(json.dumps(reports, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="Validate ETH/SOL strategy behavior.")
    parser.add_argument("--assets", nargs="+", default=["ETH", "SOL"])
    parser.add_argument("--start", default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--end", default="2025-01-03T00:00:00+00:00")
    args = parser.parse_args()
    args.start = datetime.fromisoformat(args.start)
    args.end = datetime.fromisoformat(args.end)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

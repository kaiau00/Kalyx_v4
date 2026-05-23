#!/usr/bin/env python3
"""
Run backtests across three distinct market regimes and report results.

Usage:
    python scripts/run_backtests.py

The three regimes are:
    1. Bull  (Q1–Q2 2024)       — strong upward trend
    2. Bear  (Q3–Q4 2024)      — sharp downward trend
    3. Sideways (Q4 2024–Q1 2025) — low-vol range

Each regime runs the full KalshiBacktester over its date range and prints
a summary table including win rate, avg edge, Sharpe, and max drawdown.
Per-signal attribution is printed separately.
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtesting.backtest_engine import BacktestConfig, KalshiBacktester, run_regime_backtest
from core.backtesting.historical_data import HistoricalDataLoader


REGIMES = {
    "bull_2024": {
        "label": "2024 Bull (Q1–Q2)",
        "start": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "end": datetime(2024, 6, 30, tzinfo=timezone.utc),
    },
    "bear_2024": {
        "label": "2024 Bear (Q3–Q4)",
        "start": datetime(2024, 7, 1, tzinfo=timezone.utc),
        "end": datetime(2024, 12, 31, tzinfo=timezone.utc),
    },
    "sideways_2024_2025": {
        "label": "Sideways (Q4 2024–Q1 2025)",
        "start": datetime(2024, 10, 1, tzinfo=timezone.utc),
        "end": datetime(2025, 3, 31, tzinfo=timezone.utc),
    },
}


def build_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        slippage_cents=args.slippage,
        fee_per_contract=args.fee,
        synthetic_spread_cents=args.spread,
        synthetic_top_depth=args.depth,
        bankroll=args.bankroll,
        kelly_multiplier=args.kelly,
        max_trade_fraction=args.max_trade_fraction,
        max_trade_dollars=args.max_trade_dollars,
        max_daily_loss=args.max_daily_loss,
        max_total_exposure=args.max_exposure,
        ev_threshold=args.ev_threshold,
        min_edge_after_fees=args.min_edge,
        dry_run=True,
    )


async def run_all_regimes(args: argparse.Namespace) -> dict:
    config = build_config(args)
    results = {}

    for key, meta in REGIMES.items():
        label = meta["label"]
        start = meta["start"]
        end = meta["end"]

        print(f"\n{'='*60}")
        print(f"Running backtest: {label}")
        print(f"  {start.date()} → {end.date()}")
        print(f"{'='*60}")

        try:
            result = await run_regime_backtest(
                regime_label=key,
                start=start,
                end=end,
                config=config,
            )
            results[key] = result

            print(f"\n  ── Summary ──────────────────────────────────────────")
            print(f"  Total trades    : {result.total_trades}")
            print(f"  Winning trades  : {result.winning_trades}")
            print(f"  Losing trades   : {result.losing_trades}")
            print(f"  Win rate        : {result.win_rate:.1%}")
            print(f"  Avg edge        : {result.avg_edge:+.4f}")
            print(f"  Avg edge (winners)  : {result.avg_edge_winners:+.4f}")
            print(f"  Avg edge (losers)   : {result.avg_edge_losers:+.4f}")
            print(f"  Total PnL       : ${result.total_pnl:.2f}")
            print(f"  Max drawdown    : ${result.max_drawdown:.2f}")
            print(f"  Sharpe (ann.)   : {result.sharpe_annualized:+.3f}")
            print(f"\n  ── Per-Signal Accuracy ────────────────────────────────")
            for name, acc in sorted(result.per_signal_accuracy.items()):
                cnt = result.per_signal_count.get(name, 0)
                avg_e = result.per_signal_avg_edge.get(name, 0.0)
                print(f"  {name:<30} acc={acc:.1%}  edge={avg_e:+.4f}  n={cnt}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[key] = None

    return results


def print_overall_summary(results: dict) -> None:
    print(f"\n\n{'#'*60}")
    print("#  OVERALL COMPARISON ACROSS REGIMES")
    print(f"{'#'*60}")
    print(f"\n{'Regime':<30} {'Trades':>7} {'WinRate':>8} {'AvgEdge':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 75)
    for key, result in results.items():
        if result is None:
            print(f"  {key:<28}  ERROR")
            continue
        r = result
        print(
            f"  {key:<28} {r.total_trades:>7} "
            f"{r.win_rate:>7.1%} {r.avg_edge:>+8.4f} "
            f"{r.sharpe_annualized:>+8.3f} ${r.max_drawdown:>7.2f}"
        )


def save_results(results: dict, out_path: Path) -> None:
    payload = {}
    for key, result in results.items():
        if result is not None:
            payload[key] = result.to_dict()
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nResults saved to {out_path}")


async def async_main(args: argparse.Namespace) -> None:
    if args.regime:
        REGIMES_CHOICES = {k: v for k, v in REGIMES.items() if k in args.regime}
    else:
        REGIMES_CHOICES = REGIMES

    results = {}
    for key, meta in REGIMES_CHOICES.items():
        label = meta["label"]
        start = meta["start"]
        end = meta["end"]
        config = build_config(args)

        print(f"\n{'='*60}")
        print(f"Running backtest: {label} ({start.date()} → {end.date()})")
        print(f"{'='*60}")

        try:
            loader = HistoricalDataLoader()
            windows = await loader.load_windows(start, end)
            await loader.close()

            if not windows:
                print(f"  ERROR: No backtest windows loaded.")
                results[key] = None
                continue

            print(f"  Loaded {len(windows)} windows")

            tester = KalshiBacktester(windows, config=config, regime=key)
            result = await tester.run()
            results[key] = result

            print(f"\n  Summary:")
            print(f"    Trades      : {result.total_trades}")
            print(f"    Win rate    : {result.win_rate:.1%}")
            print(f"    Avg edge    : {result.avg_edge:+.4f}")
            print(f"    Total PnL  : ${result.total_pnl:.2f}")
            print(f"    Max DD      : ${result.max_drawdown:.2f}")
            print(f"    Sharpe     : {result.sharpe_annualized:+.3f}")
            print(f"    Data source : {result.data_source} (synthetic={result.synthetic_used})")
            print(f"\n  Per-Signal:")
            for name, acc in sorted(result.per_signal_accuracy.items()):
                cnt = result.per_signal_count.get(name, 0)
                avg_e = result.per_signal_avg_edge.get(name, 0.0)
                print(f"    {name:<28} acc={acc:.1%}  edge={avg_e:+.4f}  n={cnt}")

        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            results[key] = None

    print_overall_summary(results)
    if args.output:
        save_results(results, Path(args.output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kalshi backtests across market regimes.")
    parser.add_argument("--regime", nargs="*", choices=list(REGIMES), default=None,
                      help="Regimes to run (default: all three)")
    parser.add_argument("--slippage", type=float, default=0.5, help="Slippage in cents (default: 0.5)")
    parser.add_argument("--fee", type=float, default=0.01, help="Fee per contract (default: 0.01)")
    parser.add_argument("--spread", type=float, default=3.0, help="Synthetic spread in cents (default: 3.0)")
    parser.add_argument("--depth", type=float, default=10.0, help="Synthetic top-level depth (default: 10.0)")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Simulated bankroll (default: 1000)")
    parser.add_argument("--kelly", type=float, default=0.50, help="Kelly multiplier (default: 0.50)")
    parser.add_argument("--max-trade-fraction", type=float, default=0.25)
    parser.add_argument("--max-trade-dollars", type=float, default=25.0)
    parser.add_argument("--max-daily-loss", type=float, default=50.0)
    parser.add_argument("--max-exposure", type=float, default=100.0)
    parser.add_argument("--ev-threshold", type=float, default=0.01)
    parser.add_argument("--min-edge", type=float, default=0.005)
    parser.add_argument("--output", default="backtest_results.json", help="Output JSON path")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

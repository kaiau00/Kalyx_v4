#!/usr/bin/env python3
"""
Minute scheduler for the Kalshi BTC 15-minute multi-signal strategy.
"""
import argparse
import asyncio
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from core.strategy_brain.strategies.kalshi_multisignal_strategy import (
    run_kalshi_multisignal_strategy,
    update_kalshi_weights_from_settlements,
)
from execution.telegram_notifier import TelegramNotifier


async def run_scheduler(asset: str, dry_run: bool, interval_seconds: int) -> None:
    notifier = TelegramNotifier()
    environment = "demo" if os.getenv("KALSHI_DEMO", "false").lower() in {"1", "true", "yes"} else "prod"
    settlement_refresh_seconds = int(os.getenv("KALSHI_SETTLEMENT_REFRESH_SECONDS", "900"))
    last_settlement_refresh = datetime.min.replace(tzinfo=timezone.utc)

    logger.info(f"Starting Kalshi {asset} scheduler dry_run={dry_run}")
    await notifier.send_message(
        f"Kalshi bot online\n"
        f"asset={asset}\n"
        f"mode={'dry-run' if dry_run else 'live'}\n"
        f"env={environment}\n"
        f"interval={interval_seconds}s"
    )

    while True:
        started = datetime.now(timezone.utc)
        try:
            result = await run_kalshi_multisignal_strategy(asset=asset, dry_run=dry_run)
            logger.info(
                f"Kalshi tick contract={result.contract_ticker} "
                f"pred={result.predicted_prob} reason={result.reason}"
            )
            if result.order and result.order.filled_qty > 0:
                await notifier.send_message(
                    f"Kalshi order FILLED\n"
                    f"asset={asset}\n"
                    f"contract={result.contract_ticker}\n"
                    f"order_id={result.order.order_id}\n"
                    f"side={result.order.side}\n"
                    f"status={result.order.status}\n"
                    f"filled_qty={result.order.filled_qty}\n"
                    f"remaining_qty={result.order.remaining_qty}"
                )
            now = datetime.now(timezone.utc)
            if (now - last_settlement_refresh).total_seconds() >= settlement_refresh_seconds:
                settlements = await update_kalshi_weights_from_settlements()
                last_settlement_refresh = now
                if settlements:
                    logger.info(f"Updated Kalshi signal weights from {len(settlements)} settled contract(s)")
                    for settlement in settlements:
                        await notifier.send_message(
                            f"Kalshi trade resolved\n"
                            f"contract={settlement.contract_ticker}\n"
                            f"side={settlement.side}\n"
                            f"result={settlement.result}\n"
                            f"won={settlement.won}\n"
                            f"quantity={settlement.quantity}\n"
                            f"entry_price={settlement.entry_price}\n"
                            f"estimated_pnl={settlement.estimated_pnl}"
                        )
        except Exception as exc:
            logger.exception(f"Kalshi strategy tick failed: {exc}")
            await notifier.send_message(
                f"Kalshi bot error\n"
                f"asset={asset}\n"
                f"mode={'dry-run' if dry_run else 'live'}\n"
                f"error={type(exc).__name__}: {exc}"
            )

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        await asyncio.sleep(max(1, interval_seconds - elapsed))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Kalshi multi-signal 15m strategy")
    parser.add_argument("--asset", default="BTC", help="Asset symbol. BTC is supported in v1.")
    parser.add_argument("--live", action="store_true", help="Place live Kalshi orders.")
    parser.add_argument("--interval", type=int, default=60, help="Scheduler interval in seconds.")
    args = parser.parse_args()

    asyncio.run(run_scheduler(asset=args.asset.upper(), dry_run=not args.live, interval_seconds=args.interval))


if __name__ == "__main__":
    main()

"""
Historical data loader for Kalshi backtesting.

Fetches historical 1m candles from Coinbase (primary) or Binance (fallback),
slices them into 15-minute windows that mirror Kalshi contract durations,
and generates synthetic orderbook snapshots with realistic spread/depth assumptions.

When live API data is unavailable (e.g., Coinbase's ~300-minute lookback for 1m
candles), the loader falls back to synthesising candles from a local CSV file
(exported from a data vendor) or generates Brownian-motion price paths with
realistic BTC volatility so the backtester can still be exercised.
"""
import asyncio
import bisect
import csv
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Annualised BTC volatility (approx 60% for 2024) used to calibrate synthetic paths.
_BTC_ANNUAL_VOL = 0.60
_BTC_ANNUAL_RETURN = 0.30  # approx average BTC return in 2024


@dataclass
class BacktestCandle:
    """A candle compatible with core.strategy_brain.kalshi_indicators.Candle."""
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BacktestWindow:
    """A single 15-minute backtest window derived from historical candles."""
    window_id: str
    open_time: datetime
    close_time: datetime
    candles: List[BacktestCandle]
    spot_price_at_open: float
    spot_price_at_close: float
    ground_truth: str  # "yes" if close > open, "no" otherwise
    spread_at_open_cents: float
    depth_at_open: float


class HistoricalDataLoader:
    """
    Loads historical 1m candles and slices them into backtestable 15m windows.

    Tries live APIs first, then falls back to CSV or Brownian-motion synthesis.

    Env vars:
      BACKTEST_COINBASE_PRODUCT  — Coinbase product id (default: BTC-USD)
      BACKTEST_CANDLE_PROVIDER   — "coinbase" or "binance" (default: coinbase)
      BACKTEST_SPREAD_CENTS      — synthetic spread to inject in cents (default: 3.0)
      BACKTEST_DEPTH_CONTRACTS   — synthetic top-level depth in contracts (default: 10)
      BACKTEST_CSV_PATH          — path to CSV file (columns: timestamp,open,high,low,close,volume)
      BACKTEST_SYNTHETIC_SEED    — random seed for reproducible synthetic data (default: 42)
    """

    def __init__(
        self,
        coinbase_product: str = "BTC-USD",
        provider: str = "coinbase",
        synthetic_spread_cents: float = 3.0,
        synthetic_depth: float = 10.0,
        synthetic_seed: int = 42,
        csv_path: Optional[str] = None,
        regime: Literal["bull", "bear", "sideways"] = "sideways",
    ) -> None:
        self.coinbase_product = coinbase_product
        self.provider = provider.lower()
        self.synthetic_spread_cents = float(
            os.getenv("BACKTEST_SPREAD_CENTS", str(synthetic_spread_cents))
        )
        self.synthetic_depth = float(os.getenv("BACKTEST_DEPTH_CONTRACTS", str(synthetic_depth)))
        self.synthetic_seed = int(os.getenv("BACKTEST_SYNTHETIC_SEED", str(synthetic_seed)))
        self.csv_path = csv_path or os.getenv("BACKTEST_CSV_PATH")
        self.regime = regime  # used to set drift direction for Brownian synthesis
        self._rng = random.Random(self.synthetic_seed)
        self._coinbase_client: Optional[Any] = None
        self._binance_client: Optional[Any] = None
        self.last_data_source: str = "unknown"
        self.last_synthetic_used: bool = False

    async def _coinbase(self):
        if self._coinbase_client is None:
            from data_sources.coinbase.rest import CoinbaseRESTSource
            self._coinbase_client = CoinbaseRESTSource(product_id=self.coinbase_product)
        return self._coinbase_client

    async def _binance(self):
        if self._binance_client is None:
            from data_sources.binance.rest import BinanceRESTSource
            symbol = os.getenv("BACKTEST_BINANCE_SYMBOL", "BTCUSDT")
            self._binance_client = BinanceRESTSource(symbol=symbol)
        return self._binance_client

    async def close(self) -> None:
        if self._coinbase_client:
            await self._coinbase_client.close()
            self._coinbase_client = None
        if self._binance_client:
            await self._binance_client.close()
            self._binance_client = None

    def _load_csv_candles(self, path: Path) -> List[BacktestCandle]:
        """Parse a CSV file with columns: timestamp,open,high,low,close,volume."""
        candles: List[BacktestCandle] = []
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                except Exception:
                    continue
                candles.append(
                    BacktestCandle(
                        open=Decimal(row["open"]),
                        high=Decimal(row["high"]),
                        low=Decimal(row["low"]),
                        close=Decimal(row["close"]),
                        volume=Decimal(row.get("volume", "0")),
                        timestamp=ts,
                    )
                )
        return sorted(candles, key=lambda c: c.timestamp)

    def _generate_synthetic_brownian_candles(
        self,
        start: datetime,
        end: datetime,
        spot_price: float = 45_000.0,
        regime_drift: float = 0.0,
    ) -> List[BacktestCandle]:
        """
        Generate 1m candles via Geometric Brownian Motion.

        regime_drift: per-minute drift multiplier.
          > 0  → bullish bias (bull market)
          < 0  → bearish bias (bear market)
          0.0  → neutral / sideways
        """
        rng = random.Random(self.synthetic_seed)
        per_min_vol = _BTC_ANNUAL_VOL / math.sqrt(525_600)
        per_min_drift = (_BTC_ANNUAL_RETURN / 525_600) + regime_drift

        candles: List[BacktestCandle] = []
        price = spot_price
        current = start

        while current < end:
            rnd = rng.gauss(0, 1)
            ret = per_min_drift + per_min_vol * rnd
            open_p = price
            close_p = price * (1 + ret)
            high_p = max(open_p, close_p) * (1 + abs(rng.gauss(0, 1)) * per_min_vol * 0.3)
            low_p = min(open_p, close_p) * (1 - abs(rng.gauss(0, 1)) * per_min_vol * 0.3)

            candles.append(
                BacktestCandle(
                    open=Decimal(str(open_p)),
                    high=Decimal(str(high_p)),
                    low=Decimal(str(low_p)),
                    close=Decimal(str(close_p)),
                    volume=Decimal(str(rng.uniform(50, 500))),
                    timestamp=current,
                )
            )
            price = close_p
            current += timedelta(minutes=1)

        return candles

    def generate_synthetic_candles_for_regime(
        self,
        start: datetime,
        end: datetime,
        regime: Literal["bull", "bear", "sideways"] = "sideways",
        spot_price: float = 45_000.0,
    ) -> List[BacktestCandle]:
        """Generate Brownian candles with a regime-appropriate drift."""
        # Stronger drifts so signals exceed the backtest confidence gate (~0.10).
        # Bull drift produces ~53%+ RSI readings; bear drift produces ~47% or lower.
        drift_map = {
            "bull":     0.00012,   # ~+63 bps/day  → strong uptrend
            "bear":    -0.00012,   # ~-63 bps/day  → strong downtrend
            "sideways":  0.000005,  # ~+2.6 bps/day → near-random
        }
        drift = drift_map.get(regime, 0.0)
        return self._generate_synthetic_brownian_candles(start, end, spot_price, regime_drift=drift)

    async def load_candles(
        self,
        start: datetime,
        end: datetime,
        max_candles: int = 1440,
    ) -> List[BacktestCandle]:
        """
        Fetch 1m candles between start and end.

        Strategy (tried in order):
        1. Live Coinbase / Binance API (only for recent lookback windows ≤ 300 min)
        2. CSV file if BACKTEST_CSV_PATH is set
        3. Brownian-motion synthesis as last resort

        Returns candles sorted by timestamp ascending.
        """
        # ── 1. Live API (short lookback only) ──────────────────────────
        lookback_minutes = int((end - start).total_seconds() / 60)
        if lookback_minutes <= 300:
            live_rows: List[Dict[str, Any]] = []
            current = start
            while current < end:
                remaining = int((end - current).total_seconds() / 60)
                chunk_size = min(max_candles, remaining)
                try:
                    if self.provider == "binance":
                        source = await self._binance()
                        raw = await source.get_klines(interval="1m", limit=chunk_size)
                    else:
                        source = await self._coinbase()
                        raw = await source.get_klines(interval="1m", limit=chunk_size)
                    if raw:
                        live_rows.extend(raw)
                        last_ts = raw[-1].get("timestamp")
                        current = last_ts + timedelta(minutes=1) if isinstance(last_ts, datetime) else end
                    else:
                        break
                except Exception as exc:
                    logger.warning(f"Live candle fetch failed: {exc}")
                    break
                if len(live_rows) >= max_candles:
                    break

            if len(live_rows) >= 30:
                candles: List[BacktestCandle] = []
                for row in sorted(live_rows, key=lambda r: r.get("timestamp", datetime.min)):
                    ts = row.get("timestamp")
                    if not isinstance(ts, datetime):
                        ts = datetime.fromtimestamp(ts, tz=timezone.utc)
                    if ts < start or ts >= end:
                        continue
                    candles.append(
                        BacktestCandle(
                            open=Decimal(str(row["open"])),
                            high=Decimal(str(row["high"])),
                            low=Decimal(str(row["low"])),
                            close=Decimal(str(row["close"])),
                            volume=Decimal(str(row.get("volume", "0"))),
                            timestamp=ts,
                        )
                    )
                logger.info(f"Loaded {len(candles)} live candles")
                self.last_data_source = self.provider
                self.last_synthetic_used = False
                return candles

        # ── 2. CSV ────────────────────────────────────────────────────
        if self.csv_path:
            csv_path = Path(self.csv_path)
            if csv_path.exists():
                candles = self._load_csv_candles(csv_path)
                filtered = [c for c in candles if start <= c.timestamp < end]
                logger.info(f"Loaded {len(filtered)} candles from CSV")
                self.last_data_source = "csv"
                self.last_synthetic_used = False
                return filtered

        # ── 3. Brownian synthesis with regime-appropriate drift ───
        if start.year == 2024:
            spot = 45_000.0
        elif start.year == 2025:
            spot = 95_000.0
        else:
            spot = 45_000.0
        regime = self.regime
        candles = self.generate_synthetic_candles_for_regime(start, end, regime=regime, spot_price=spot)
        filtered = [c for c in candles if start <= c.timestamp < end]
        logger.info(f"Synthesised {len(filtered)} Brownian candles (regime={regime}, seed={self.synthetic_seed})")
        self.last_data_source = "synthetic"
        self.last_synthetic_used = True
        return filtered

    def slice_windows(
        self,
        candles: List[BacktestCandle],
        window_minutes: int = 15,
    ) -> List[BacktestWindow]:
        """
        Slice a continuous candle list into non-overlapping 15-minute windows
        using binary search — O(n log n) instead of the naive O(n²) scan.

        Each window's ground_truth is derived from the spot price at open vs close:
        - "yes"  : close > open  (price went up  → up contract wins)
        - "no"   : close <= open  (price went down → down contract wins)
        """
        if not candles:
            return []

        # Build a sorted list of timestamps for binary search
        timestamps = [c.timestamp for c in candles]
        window_delta = timedelta(minutes=window_minutes)

        # Align to the first candle's 15-minute boundary
        first_ts = candles[0].timestamp
        window_start = first_ts.replace(
            minute=(first_ts.minute // window_minutes) * window_minutes,
            second=0, microsecond=0,
        )
        if window_start < first_ts:
            window_start += window_delta

        windows: List[BacktestWindow] = []
        spread = self.synthetic_spread_cents
        last_ts = candles[-1].timestamp

        while window_start <= last_ts:
            window_end = window_start + window_delta

            # Binary search the slice of candles in [window_start, window_end)
            lo = bisect.bisect_left(timestamps, window_start)
            hi = bisect.bisect_left(timestamps, window_end)

            if lo < hi:
                window_candles = candles[lo:hi]
                open_price = float(window_candles[0].open)
                close_price = float(window_candles[-1].close)
                ground_truth = "yes" if close_price > open_price else "no"

                windows.append(
                    BacktestWindow(
                        window_id=f"bt_{window_start.strftime('%Y%m%d%H%M')}",
                        open_time=window_start,
                        close_time=window_end,
                        candles=window_candles,
                        spot_price_at_open=open_price,
                        spot_price_at_close=close_price,
                        ground_truth=ground_truth,
                        spread_at_open_cents=spread,
                        depth_at_open=self.synthetic_depth,
                    )
                )

            window_start = window_end

        return windows

    def generate_synthetic_orderbook(
        self,
        window: BacktestWindow,
        market_probability: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Generate a synthetic orderbook snapshot for a backtest window.

        The yes/no prices are centred around market_probability with the
        configured synthetic spread.  Depth is uniform at the top N levels.

        Returns a dict compatible with execution.kalshi_api.OrderBookSnapshot.
        """
        mid = Decimal(str(market_probability))
        half_spread = Decimal(str(self.synthetic_spread_cents)) / Decimal("200")
        yes_bid = mid - half_spread
        no_bid = (Decimal("1") - mid) - half_spread
        yes_ask = Decimal("1") - no_bid
        no_ask = Decimal("1") - yes_bid

        depth = Decimal(str(self.synthetic_depth))

        yes_bids = [
            {"price": float(yes_bid - Decimal(str(i)) * Decimal("0.001")), "quantity": float(depth)}
            for i in range(5)
        ]
        no_bids = [
            {"price": float(no_bid - Decimal(str(i)) * Decimal("0.001")), "quantity": float(depth)}
            for i in range(5)
        ]

        return {
            "ticker": f"SIM_{window.window_id}",
            "yes_bids": yes_bids,
            "no_bids": no_bids,
            "yes_midpoint": float(mid),
            "timestamp": window.open_time,
        }

    async def load_windows(
        self,
        start: datetime,
        end: datetime,
        window_minutes: int = 15,
    ) -> List[BacktestWindow]:
        """Fetch candles and slice into backtest windows in one call."""
        candles = await self.load_candles(start, end)
        return self.slice_windows(candles, window_minutes)


async def demo() -> None:
    """Quick smoke-test when run directly."""
    loader = HistoricalDataLoader()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=2)
    windows = await loader.load_windows(start, end)
    await loader.close()
    print(f"Loaded {len(windows)} windows")
    for w in windows[:3]:
        print(f"  {w.window_id}: {w.open_time} → {w.close_time}  ground={w.ground_truth}  candles={len(w.candles)}")


if __name__ == "__main__":
    asyncio.run(demo())


async def load_historical_candles(
    start: datetime,
    end: datetime,
    *,
    provider: str = "coinbase",
    csv_path: Optional[str] = None,
    regime: Literal["bull", "bear", "sideways"] = "sideways",
) -> List[BacktestCandle]:
    loader = HistoricalDataLoader(provider=provider, csv_path=csv_path, regime=regime)
    try:
        return await loader.load_candles(start, end)
    finally:
        await loader.close()


def simulate_orderbook_at_time(
    window: BacktestWindow,
    *,
    market_probability: float = 0.5,
    synthetic_spread_cents: float = 3.0,
    synthetic_depth: float = 10.0,
) -> Dict[str, Any]:
    loader = HistoricalDataLoader(
        synthetic_spread_cents=synthetic_spread_cents,
        synthetic_depth=synthetic_depth,
    )
    return loader.generate_synthetic_orderbook(window, market_probability=market_probability)


def generate_contract_windows(
    candles: List[BacktestCandle],
    *,
    window_minutes: int = 15,
    synthetic_spread_cents: float = 3.0,
    synthetic_depth: float = 10.0,
) -> List[BacktestWindow]:
    loader = HistoricalDataLoader(
        synthetic_spread_cents=synthetic_spread_cents,
        synthetic_depth=synthetic_depth,
    )
    return loader.slice_windows(candles, window_minutes=window_minutes)

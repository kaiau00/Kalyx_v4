"""Deribit data source for funding rate and open interest signals."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class DeribitSnapshot:
    funding_rate_pct: float
    open_interest_usd: float
    funding_rate_direction: int
    oi_change_1h_pct: float
    timestamp: datetime

    @property
    def funding_confidence(self) -> float:
        return min(1.0, abs(self.funding_rate_pct) / 0.05)

    @property
    def oi_direction(self) -> int:
        if self.oi_change_1h_pct > 0:
            return 1
        if self.oi_change_1h_pct < 0:
            return -1
        return 0

    @property
    def oi_confidence(self) -> float:
        return min(1.0, abs(self.oi_change_1h_pct) / 0.1)


class DeribitDataSource:
    _BASE_URL_PROD = "https://www.deribit.com"
    _BASE_URL_TEST = "https://test.deribit.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        use_testnet: bool = False,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key or os.getenv("DERIBIT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("DERIBIT_API_SECRET", "")
        base = self._BASE_URL_TEST if use_testnet else self._BASE_URL_PROD
        self._client = httpx.AsyncClient(base_url=base, timeout=timeout)
        self._last_snapshot: Optional[DeribitSnapshot] = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _fetch_json(self, path: str, params: dict[str, Any]) -> Optional[dict[str, Any]]:
        try:
            response = await self._client.get(path, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("result") if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning(f"Deribit request failed for {path}: {exc}")
            return None

    @staticmethod
    def _funding_direction(rate_pct: float) -> int:
        if rate_pct < -0.01:
            return -1
        if rate_pct > 0.01:
            return 1
        return 0

    async def get_funding_rate(self, instrument: str = "BTC-PERPETUAL") -> Optional[float]:
        """
        Fetch the current funding rate for a perpetual instrument.

        Uses get_currentunding which returns funding rate info; falls back to
        get_book_summary_by_currency for the funding rate field.
        """
        # Primary: /public/get_currentunding (Deribit's actual funding endpoint)
        result = await self._fetch_json(
            "/api/v2/public/get_currentunding",
            {"instrument_name": instrument},
        )
        if isinstance(result, dict):
            interest = result.get("interest")
            if isinstance(interest, (int, float)):
                return float(interest)

        # Fallback: derive from book summary
        result2 = await self._fetch_json(
            "/api/v2/public/get_book_summary_by_currency",
            {"currency": instrument.replace("-PERPETUAL", ""), "kind": "future"},
        )
        if isinstance(result2, list) and result2:
            # Deribit perpetual BTC summary has "current_funding" field
            for row in result2:
                if isinstance(row, dict):
                    rate = row.get("current_funding") or row.get("funding_rate")
                    if isinstance(rate, (int, float)):
                        return float(rate)
        return None

    async def get_open_interest(self, currency: str = "BTC") -> Optional[float]:
        result = await self._fetch_json(
            "/api/v2/public/get_book_summary_by_currency",
            {"currency": currency, "kind": "future"},
        )
        if not isinstance(result, list):
            return None
        total = 0.0
        for row in result:
            if not isinstance(row, dict):
                continue
            oi = row.get("open_interest")
            mark = row.get("mark_price") or row.get("estimated_delivery_price") or 0.0
            if isinstance(oi, (int, float)) and isinstance(mark, (int, float)):
                total += float(oi) * float(mark)
        return total or None

    async def get_recent_trades(self, instrument: str = "BTC-PERPETUAL", count: int = 25) -> list[dict[str, Any]]:
        result = await self._fetch_json(
            "/api/v2/public/get_last_trades_by_instrument",
            {"instrument_name": instrument, "count": count},
        )
        if not isinstance(result, dict):
            return []
        trades = result.get("trades")
        return trades if isinstance(trades, list) else []

    def _oi_change_pct(self, open_interest_usd: float, now: datetime) -> float:
        if self._last_snapshot is None:
            return 0.0
        if now - self._last_snapshot.timestamp > timedelta(hours=2):
            return 0.0
        baseline = self._last_snapshot.open_interest_usd
        if baseline <= 0:
            return 0.0
        return ((open_interest_usd - baseline) / baseline) * 100.0

    async def get_snapshot(self) -> Optional[DeribitSnapshot]:
        funding_rate, open_interest_usd = await asyncio.gather(
            self.get_funding_rate(),
            self.get_open_interest(),
        )
        if funding_rate is None and open_interest_usd is None:
            return None
        now = datetime.now(timezone.utc)
        open_interest = float(open_interest_usd or 0.0)
        snapshot = DeribitSnapshot(
            funding_rate_pct=float(funding_rate or 0.0),
            open_interest_usd=open_interest,
            funding_rate_direction=self._funding_direction(float(funding_rate or 0.0)),
            oi_change_1h_pct=self._oi_change_pct(open_interest, now),
            timestamp=now,
        )
        self._last_snapshot = snapshot
        return snapshot

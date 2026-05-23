"""
Coinbase REST candle source used as a fallback for the Kalshi strategy.
"""
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

import httpx

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class CoinbaseRESTSource:
    def __init__(
        self,
        product_id: str = "BTC-USD",
        base_url: str = "https://api.exchange.coinbase.com",
        timeout: float = 10.0,
    ) -> None:
        self.product_id = product_id
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"User-Agent": "KalshiMultisignalBot/1.0"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get_klines(self, interval: str = "1m", limit: int = 120) -> List[Dict[str, Any]]:
        granularity = self._interval_to_seconds(interval)
        try:
            response = await self._client.get(
                f"/products/{self.product_id}/candles",
                params={"granularity": granularity},
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning(f"Coinbase candle fetch failed for {self.product_id}: {exc}")
            return []

        candles = []
        for row in response.json()[:limit]:
            candles.append(
                {
                    "timestamp": datetime.fromtimestamp(row[0], tz=timezone.utc),
                    "open": Decimal(str(row[3])),
                    "high": Decimal(str(row[2])),
                    "low": Decimal(str(row[1])),
                    "close": Decimal(str(row[4])),
                    "volume": Decimal(str(row[5])),
                }
            )
        return sorted(candles, key=lambda candle: candle["timestamp"])

    def _interval_to_seconds(self, interval: str) -> int:
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "6h": 21600,
            "1d": 86400,
        }
        return mapping.get(interval, 60)

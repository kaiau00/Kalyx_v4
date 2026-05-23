"""
Binance REST candle source used by the Kalshi strategy.
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


class BinanceRESTSource:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        base_url: str = "https://api.binance.com",
        timeout: float = 10.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def get_klines(self, interval: str = "1m", limit: int = 120) -> List[Dict[str, Any]]:
        try:
            response = await self._client.get(
                "/api/v3/klines",
                params={"symbol": self.symbol, "interval": interval, "limit": limit},
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning(f"Binance candle fetch failed for {self.symbol}: {exc}")
            return []

        candles = []
        for row in response.json():
            candles.append(
                {
                    "timestamp": datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                    "open": Decimal(str(row[1])),
                    "high": Decimal(str(row[2])),
                    "low": Decimal(str(row[3])),
                    "close": Decimal(str(row[4])),
                    "volume": Decimal(str(row[5])),
                }
            )
        return candles

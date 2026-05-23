"""
Kalshi REST API connector for 15-minute crypto up/down markets.

The connector intentionally keeps request signing and response parsing local so
strategy code can depend on typed objects instead of raw API payloads.
"""
import asyncio
import base64
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


OrderSide = Literal["yes", "no"]


@dataclass
class ContractRef:
    ticker: str
    event_ticker: Optional[str]
    title: str
    open_time: Optional[datetime]
    close_time: Optional[datetime]
    status: str
    yes_bid: Optional[Decimal] = None
    yes_ask: Optional[Decimal] = None
    no_bid: Optional[Decimal] = None
    no_ask: Optional[Decimal] = None
    open_spread_cents: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBookLevel:
    price: Decimal
    quantity: Decimal


@dataclass
class OrderBookSnapshot:
    ticker: str
    yes_bids: List[OrderBookLevel]
    no_bids: List[OrderBookLevel]
    timestamp: datetime

    @property
    def best_yes_bid(self) -> Optional[Decimal]:
        return self.yes_bids[0].price if self.yes_bids else None

    @property
    def best_no_bid(self) -> Optional[Decimal]:
        return self.no_bids[0].price if self.no_bids else None

    @property
    def best_yes_ask(self) -> Optional[Decimal]:
        if not self.no_bids:
            return None
        return Decimal("1") - self.no_bids[0].price

    @property
    def best_no_ask(self) -> Optional[Decimal]:
        if not self.yes_bids:
            return None
        return Decimal("1") - self.yes_bids[0].price

    @property
    def yes_midpoint(self) -> Optional[Decimal]:
        if self.yes_bids and self.no_bids:
            derived_yes_ask = Decimal("1") - self.no_bids[0].price
            derived_no_ask = Decimal("1") - self.yes_bids[0].price
            yes_mid = (self.yes_bids[0].price + derived_yes_ask) / 2
            no_mid = (self.no_bids[0].price + derived_no_ask) / 2
            return (yes_mid + (Decimal("1") - no_mid)) / 2
        return None


@dataclass
class BestMarket:
    ticker: str
    yes_bid: Optional[Decimal]
    yes_ask: Optional[Decimal]
    no_bid: Optional[Decimal]
    no_ask: Optional[Decimal]


@dataclass
class OrderResult:
    order_id: str
    client_order_id: str
    ticker: str
    side: OrderSide
    status: str
    filled_qty: Decimal = Decimal("0")
    remaining_qty: Decimal = Decimal("0")
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderStatusResult:
    order_id: str
    status: str
    filled_quantity: Decimal
    remaining_quantity: Decimal
    raw: Dict[str, Any]


@dataclass
class PositionSnapshot:
    ticker: str
    position: Decimal
    market_exposure: Decimal
    raw: Dict[str, Any]


@dataclass
class SettlementResult:
    ticker: str
    result: Optional[str]
    settled_time: Optional[datetime]
    raw: Dict[str, Any]


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _dec(value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    return Decimal(str(value))


def _price_dec(value: Any) -> Optional[Decimal]:
    raw = _dec(value)
    if raw is None:
        return None
    if raw > 1:
        return raw / Decimal("100")
    return raw


class KalshiAPI:
    """
    Small async wrapper around Kalshi REST.

    Env vars:
      KALSHI_API_KEY or KALSHI_API_KEY_ID: API key id.
      KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH: RSA private key PEM.
      KALSHI_BASE_URL: optional full base URL ending in /trade-api/v2.
      KALSHI_DEMO=true: use demo base URL when KALSHI_BASE_URL is not set.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        private_key_path: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key or os.getenv("KALSHI_API_KEY") or os.getenv("KALSHI_API_KEY_ID")
        self.base_url = (base_url or os.getenv("KALSHI_BASE_URL") or self._default_base_url()).rstrip("/")
        self._private_key_pem = private_key_pem or os.getenv("KALSHI_PRIVATE_KEY")
        self._private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        self._private_key = None
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._active_contract_cache: Dict[str, Dict[str, Any]] = {}

    def _default_base_url(self) -> str:
        if os.getenv("KALSHI_DEMO", "").lower() in {"1", "true", "yes"}:
            return "https://external-api.demo.kalshi.co/trade-api/v2"
        return "https://external-api.kalshi.com/trade-api/v2"

    async def close(self) -> None:
        await self._client.aclose()

    def _load_private_key(self):
        if self._private_key:
            return self._private_key

        pem: bytes
        if self._private_key_pem:
            key_text = self._private_key_pem.replace("\\n", "\n").strip()
            pem = self._coerce_private_key_text(key_text)
            if self._private_key:
                return self._private_key
        elif self._private_key_path:
            with open(self._private_key_path, "rb") as key_file:
                pem = key_file.read()
        else:
            raise RuntimeError("KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH is required")

        self._private_key = serialization.load_pem_private_key(
            pem,
            password=None,
            backend=default_backend(),
        )
        return self._private_key

    def _coerce_private_key_text(self, key_text: str) -> bytes:
        if "BEGIN" in key_text:
            return key_text.encode("utf-8")

        compact = "".join(key_text.split())
        try:
            der = base64.b64decode(compact, validate=True)
            if b"BEGIN" in der and b"PRIVATE" in der:
                return der
            try:
                self._private_key = serialization.load_der_private_key(
                    der,
                    password=None,
                    backend=default_backend(),
                )
                return b""
            except Exception:
                pass
        except Exception:
            pass

        lines = [compact[i : i + 64] for i in range(0, len(compact), 64)]
        for label in ("PRIVATE KEY", "RSA PRIVATE KEY"):
            candidate = (
                f"-----BEGIN {label}-----\n"
                + "\n".join(lines)
                + f"\n-----END {label}-----\n"
            ).encode("utf-8")
            try:
                serialization.load_pem_private_key(candidate, password=None, backend=default_backend())
                return candidate
            except Exception:
                continue
        return key_text.encode("utf-8")

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError("KALSHI_API_KEY or KALSHI_API_KEY_ID is required")

        timestamp = str(int(time.time() * 1000))
        full_path = urlparse(self.base_url + path).path
        path_without_query = full_path.split("?")[0]
        message = f"{timestamp}{method.upper()}{path_without_query}".encode("utf-8")
        signature = self._load_private_key().sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        headers = self._auth_headers(method, path) if auth else {"Content-Type": "application/json"}
        max_attempts = int(os.getenv("KALSHI_HTTP_MAX_ATTEMPTS", "3"))
        base_delay = float(os.getenv("KALSHI_HTTP_RETRY_BASE_DELAY", "2.0"))

        for attempt in range(1, max_attempts + 1):
            response = await self._client.request(method, path, params=params, json=json, headers=headers)
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()

            if attempt == max_attempts:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else base_delay * attempt
            except ValueError:
                delay = base_delay * attempt
            logger.warning(
                f"Kalshi rate limit hit on {method} {path}; retrying in {delay:.1f}s "
                f"(attempt {attempt}/{max_attempts})"
            )
            await asyncio.sleep(delay)

        raise RuntimeError(f"Unreachable retry loop for {method} {path}")

    async def get_markets(
        self,
        *,
        series_ticker: Optional[str] = None,
        status: Optional[str] = "open",
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        min_settled_ts: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if max_close_ts is not None:
            params["max_close_ts"] = max_close_ts
        if min_settled_ts is not None:
            params["min_settled_ts"] = min_settled_ts

        data = await self._request("GET", "/markets", params=params)
        return data.get("markets", [])

    async def get_active_15m_contract(
        self,
        asset: str = "BTC",
        now: Optional[datetime] = None,
    ) -> Optional[ContractRef]:
        now = now or datetime.now(timezone.utc)
        config_prefix = os.getenv(f"KALSHI_{asset.upper()}_TICKER_PREFIX")
        series_ticker = os.getenv(f"KALSHI_{asset.upper()}_SERIES_TICKER")
        allow_keyword_discovery = os.getenv("KALSHI_ALLOW_KEYWORD_DISCOVERY", "false").lower() in {"1", "true", "yes"}
        keywords = [
            k.strip().lower()
            for k in os.getenv(f"KALSHI_{asset.upper()}_KEYWORDS", f"{asset},Bitcoin").split(",")
            if k.strip()
        ]
        target_minutes = int(os.getenv("KALSHI_TARGET_CONTRACT_MINUTES", "15"))
        duration_tolerance = int(os.getenv("KALSHI_CONTRACT_DURATION_TOLERANCE_MINUTES", "3"))

        if not series_ticker and not config_prefix and not allow_keyword_discovery:
            logger.warning(
                f"No explicit Kalshi market selector configured for {asset}. "
                "Set KALSHI_<ASSET>_SERIES_TICKER or KALSHI_<ASSET>_TICKER_PREFIX, "
                "or explicitly enable KALSHI_ALLOW_KEYWORD_DISCOVERY=true."
            )
            return None

        cache_bucket = int(now.timestamp()) // 900
        cache_key = f"{asset.upper()}:{cache_bucket}"
        cached = self._active_contract_cache.get(cache_key)
        if cached and cached.get("expires_at", 0) > time.time():
            return cached.get("contract")

        markets = await self.get_markets(
            series_ticker=series_ticker,
            status=os.getenv("KALSHI_MARKET_STATUS", "open"),
            min_close_ts=int(now.timestamp()) - 60,
            max_close_ts=int(now.timestamp()) + 1800,
            limit=int(os.getenv("KALSHI_MARKET_DISCOVERY_LIMIT", "100")),
        )

        candidates: List[ContractRef] = []
        for market in markets:
            ticker = str(market.get("ticker", ""))
            title = " ".join(
                str(market.get(key, ""))
                for key in ("title", "yes_sub_title", "no_sub_title", "event_ticker")
            )
            if config_prefix and not ticker.startswith(config_prefix):
                continue
            if not config_prefix and keywords and not any(k in title.lower() or k in ticker.lower() for k in keywords):
                continue

            open_time = _parse_dt(market.get("open_time"))
            close_time = _parse_dt(market.get("close_time"))
            if open_time and now < open_time:
                continue
            if close_time and now > close_time:
                continue
            if open_time and close_time:
                duration_minutes = (close_time - open_time).total_seconds() / 60.0
                if abs(duration_minutes - target_minutes) > duration_tolerance:
                    continue

            candidates.append(self._contract_from_market(market))

        candidates.sort(key=lambda c: c.close_time or datetime.max.replace(tzinfo=timezone.utc))
        contract = candidates[0] if candidates else None
        self._active_contract_cache[cache_key] = {
            "contract": contract,
            "expires_at": time.time() + float(os.getenv("KALSHI_ACTIVE_CONTRACT_CACHE_SECONDS", "45")),
        }
        stale_keys = [key for key in self._active_contract_cache if key != cache_key]
        for key in stale_keys:
            self._active_contract_cache.pop(key, None)
        return contract

    def _contract_from_market(self, market: Dict[str, Any]) -> ContractRef:
        return ContractRef(
            ticker=str(market.get("ticker", "")),
            event_ticker=market.get("event_ticker"),
            title=str(market.get("title") or market.get("yes_sub_title") or ""),
            open_time=_parse_dt(market.get("open_time")),
            close_time=_parse_dt(market.get("close_time")),
            status=str(market.get("status", "")),
            yes_bid=_price_dec(market.get("yes_bid_dollars")),
            yes_ask=_price_dec(market.get("yes_ask_dollars")),
            no_bid=_price_dec(market.get("no_bid_dollars")),
            no_ask=_price_dec(market.get("no_ask_dollars")),
            raw=market,
        )

    async def get_orderbook(self, contract_code: str, depth: int = 20) -> OrderBookSnapshot:
        data = await self._request(
            "GET",
            f"/markets/{contract_code}/orderbook",
            params={"depth": depth},
        )
        book = data.get("orderbook_fp") or data.get("orderbook") or {}
        return OrderBookSnapshot(
            ticker=contract_code,
            yes_bids=self._parse_book_levels(book.get("yes_dollars") or book.get("yes") or []),
            no_bids=self._parse_book_levels(book.get("no_dollars") or book.get("no") or []),
            timestamp=datetime.now(timezone.utc),
        )

    def _parse_book_levels(self, levels: List[List[Any]]) -> List[OrderBookLevel]:
        parsed = []
        for level in levels:
            if len(level) < 2:
                continue
            price = _price_dec(level[0])
            quantity = _dec(level[1])
            if price is None or quantity is None:
                continue
            parsed.append(OrderBookLevel(price=price, quantity=quantity))
        return sorted(parsed, key=lambda level: level.price, reverse=True)

    async def get_best_market(self, contract_code: str) -> BestMarket:
        book = await self.get_orderbook(contract_code, depth=1)
        return BestMarket(
            ticker=contract_code,
            yes_bid=book.best_yes_bid,
            yes_ask=book.best_yes_ask,
            no_bid=book.best_no_bid,
            no_ask=book.best_no_ask,
        )

    async def place_limit_order(
        self,
        contract_code: str,
        side: OrderSide,
        price_cents: int,
        quantity: int,
        client_order_id: str,
    ) -> OrderResult:
        if side not in ("yes", "no"):
            raise ValueError("side must be 'yes' or 'no'")
        # Defensive clamp: Kalshi prices are in dollars (0.01-0.99) but our
        # pipeline may accidentally pass dollar-scale values.  Clamp to the
        # valid range rather than crashing so the trade is silently skipped.
        price_cents = max(1, min(99, price_cents))
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        payload: Dict[str, Any] = {
            "ticker": contract_code,
            "side": side,
            "action": "buy",
            "client_order_id": client_order_id,
            "count": quantity,
            "time_in_force": os.getenv("KALSHI_TIME_IN_FORCE", "good_till_canceled"),
        }
        payload[f"{side}_price"] = price_cents

        data = await self._request("POST", "/portfolio/orders", json=payload, auth=True)
        order = data.get("order", data)
        return OrderResult(
            order_id=str(order.get("order_id", "")),
            client_order_id=str(order.get("client_order_id", client_order_id)),
            ticker=str(order.get("ticker", contract_code)),
            side=side,
            status=str(order.get("status", "")),
            filled_qty=Decimal(str(order.get("fill_count_fp", order.get("filled_quantity", "0")))),
            remaining_qty=Decimal(str(order.get("remaining_count_fp", order.get("remaining_quantity", "0")))),
            raw=order,
        )

    async def get_order_status(self, order_id: str) -> OrderStatusResult:
        data = await self._request("GET", f"/portfolio/orders/{order_id}", auth=True)
        order = data.get("order", data)
        return OrderStatusResult(
            order_id=str(order.get("order_id", order_id)),
            status=str(order.get("status", "")),
            filled_quantity=Decimal(str(order.get("fill_count_fp", order.get("filled_quantity", "0")))),
            remaining_quantity=Decimal(str(order.get("remaining_count_fp", order.get("remaining_quantity", "0")))),
            raw=order,
        )

    async def get_positions(self) -> List[PositionSnapshot]:
        data = await self._request("GET", "/portfolio/positions", auth=True)
        positions = data.get("market_positions") or data.get("positions") or []
        return [
            PositionSnapshot(
                ticker=str(pos.get("ticker", pos.get("market_ticker", ""))),
                position=Decimal(str(pos.get("position", pos.get("position_fp", "0")))),
                market_exposure=Decimal(str(pos.get("market_exposure", pos.get("market_exposure_dollars", "0")))),
                raw=pos,
            )
            for pos in positions
        ]

    async def get_settlements(self, start: Optional[datetime] = None) -> List[SettlementResult]:
        markets = await self.get_markets(
            status="settled",
            min_settled_ts=int(start.timestamp()) if start else None,
            limit=int(os.getenv("KALSHI_SETTLEMENT_LIMIT", "100")),
        )
        return [
            SettlementResult(
                ticker=str(market.get("ticker", "")),
                result=market.get("result"),
                settled_time=_parse_dt(market.get("settled_time") or market.get("close_time")),
                raw=market,
            )
            for market in markets
        ]

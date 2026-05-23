import asyncio
from decimal import Decimal

from execution.kalshi_api import KalshiAPI


def test_get_orderbook_parses_fixed_point_levels():
    asyncio.run(_run_orderbook_parse_test())


async def _run_orderbook_parse_test():
    async def handler(request):
        import httpx

        return httpx.Response(
            200,
            json={
                "orderbook_fp": {
                    "yes_dollars": [["0.5100", "12.00"]],
                    "no_dollars": [["0.4700", "8.00"]],
                }
            },
        )

    import httpx

    transport = httpx.MockTransport(handler)
    api = KalshiAPI(api_key="x", private_key_pem="unused", base_url="https://example.com/trade-api/v2")
    await api._client.aclose()
    api._client = httpx.AsyncClient(base_url=api.base_url, transport=transport)

    book = await api.get_orderbook("BTC-TEST")

    assert book.best_yes_bid == Decimal("0.5100")
    assert book.best_yes_ask == Decimal("0.5300")

    await api.close()


def test_request_retries_once_after_rate_limit():
    asyncio.run(_run_retry_test())


async def _run_retry_test():
    calls = {"count": 0}

    async def handler(request):
        import httpx

        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"markets": []})

    import httpx

    transport = httpx.MockTransport(handler)
    api = KalshiAPI(api_key="x", private_key_pem="unused", base_url="https://example.com/trade-api/v2")
    await api._client.aclose()
    api._client = httpx.AsyncClient(base_url=api.base_url, transport=transport)

    data = await api.get_markets(limit=1)

    assert data == []
    assert calls["count"] == 2

    await api.close()


def test_active_contract_cache_avoids_repeat_market_scan():
    asyncio.run(_run_active_contract_cache_test())


async def _run_active_contract_cache_test():
    calls = {"count": 0}

    async def handler(request):
        import httpx

        calls["count"] += 1
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "BTC-TEST-1",
                        "title": "Bitcoin",
                        "status": "open",
                        "open_time": "2026-05-22T06:00:00Z",
                        "close_time": "2026-05-22T06:15:00Z",
                        "yes_bid_dollars": "0.45",
                        "yes_ask_dollars": "0.46",
                        "no_bid_dollars": "0.53",
                        "no_ask_dollars": "0.54",
                    }
                ]
            },
        )

    import httpx
    from datetime import datetime, timezone
    import os

    transport = httpx.MockTransport(handler)
    api = KalshiAPI(api_key="x", private_key_pem="unused", base_url="https://example.com/trade-api/v2")
    await api._client.aclose()
    api._client = httpx.AsyncClient(base_url=api.base_url, transport=transport)

    os.environ["KALSHI_ALLOW_KEYWORD_DISCOVERY"] = "true"
    now = datetime(2026, 5, 22, 6, 5, tzinfo=timezone.utc)
    contract_1 = await api.get_active_15m_contract(asset="BTC", now=now)
    contract_2 = await api.get_active_15m_contract(asset="BTC", now=now)

    assert contract_1 is not None
    assert contract_2 is not None
    assert contract_1.ticker == contract_2.ticker
    assert calls["count"] == 1

    os.environ.pop("KALSHI_ALLOW_KEYWORD_DISCOVERY", None)
    await api.close()


def test_orderbook_normalizes_cent_prices_to_decimal_dollars():
    asyncio.run(_run_cent_price_normalization_test())


async def _run_cent_price_normalization_test():
    async def handler(request):
        import httpx

        return httpx.Response(
            200,
            json={
                "orderbook_fp": {
                    "yes_dollars": [["43.0000", "12.00"]],
                    "no_dollars": [["55.0000", "8.00"]],
                }
            },
        )

    import httpx

    transport = httpx.MockTransport(handler)
    api = KalshiAPI(api_key="x", private_key_pem="unused", base_url="https://example.com/trade-api/v2")
    await api._client.aclose()
    api._client = httpx.AsyncClient(base_url=api.base_url, transport=transport)

    book = await api.get_orderbook("BTC-TEST")

    assert book.best_yes_bid == Decimal("0.43")
    assert book.best_no_bid == Decimal("0.55")
    assert book.best_yes_ask == Decimal("0.45")

    await api.close()

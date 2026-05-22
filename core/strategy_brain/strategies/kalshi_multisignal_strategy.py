"""
Kalshi-native multi-signal strategy for BTC 15-minute up/down markets.
"""
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from core.strategy_brain.kalshi_fusion import FusedKalshiSignal, KalshiSignalFusion
from core.strategy_brain.kalshi_features import build_feature_snapshot
from core.strategy_brain.kalshi_indicators import (
    Candle,
    SignalValue,
    compute_indicators,
    kalshi_market_signals,
    normalize_indicators,
)
from core.strategy_brain.kalshi_model import KalshiModelPrediction, KalshiProbabilityModel
from data_sources.binance.rest import BinanceRESTSource
from data_sources.coinbase.rest import CoinbaseRESTSource
from execution.kalshi_api import KalshiAPI, OrderBookSnapshot, OrderResult
from execution.kalshi_risk import KalshiRiskConfig, TradeIntent, build_trade_intent


@dataclass
class ContractOpenState:
    ticker: str
    open_time: datetime
    yes_bid: Optional[Decimal] = None
    no_bid: Optional[Decimal] = None
    spread_at_open: Optional[float] = None
    book_imbalance_at_open: Optional[float] = None


_CONTRACT_OPEN_STATES: Dict[str, ContractOpenState] = {}


@dataclass
class StrategyRunResult:
    timestamp: datetime
    contract_ticker: Optional[str]
    predicted_prob: Optional[float]
    market_probability: Optional[float]
    trade_intent: Optional[TradeIntent]
    order: Optional[OrderResult]
    dry_run: bool
    reason: str


@dataclass
class SettlementUpdate:
    contract_ticker: str
    result: str
    side: Optional[str]
    quantity: int
    entry_price: Optional[Decimal]
    estimated_pnl: Optional[Decimal]
    won: Optional[bool]


def _decimal_env(name: str, default: str) -> Decimal:
    return Decimal(os.getenv(name, default))


def load_risk_config() -> KalshiRiskConfig:
    return KalshiRiskConfig(
        bankroll=_decimal_env("KALSHI_BANKROLL", "1000.00"),
        max_trade_fraction=_decimal_env("KALSHI_KELLY_CAP", "0.25"),
        kelly_multiplier=_decimal_env("KALSHI_KELLY_MULTIPLIER", "0.50"),
        max_trade_dollars=_decimal_env("KALSHI_MAX_TRADE_DOLLARS", "25.00"),
        max_daily_loss=_decimal_env("KALSHI_MAX_DAILY_LOSS", "50.00"),
        max_total_exposure=_decimal_env("KALSHI_MAX_TOTAL_EXPOSURE", "100.00"),
        ev_threshold=_decimal_env("KALSHI_EV_THRESHOLD", "0.01"),
        fee_per_contract=_decimal_env("KALSHI_FEE_PER_CONTRACT", "0.10"),
        min_edge_after_fees=_decimal_env("KALSHI_MIN_EDGE_AFTER_FEES", "0.03"),
        max_spread_cents=int(os.getenv("KALSHI_MAX_SPREAD_CENTS", "6")),
        min_top_depth=_decimal_env("KALSHI_MIN_TOP_DEPTH", "5"),
        min_model_confidence=_decimal_env("KALSHI_MIN_MODEL_CONFIDENCE", "0.20"),
        late_trade_cutoff_seconds=int(os.getenv("KALSHI_LATE_TRADE_CUTOFF_SECONDS", "60")),
        drawdown_size_reduction_threshold=_decimal_env("KALSHI_DRAWDOWN_SIZE_REDUCTION_THRESHOLD", "0.50"),
    )


def _candles_from_rows(rows: List[Dict[str, Any]]) -> List[Candle]:
    return [
        Candle(
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row.get("volume", Decimal("0")),
        )
        for row in rows
    ]


def _market_probability_from_book(orderbook: OrderBookSnapshot) -> Optional[float]:
    yes_ask = orderbook.best_yes_ask
    no_ask = orderbook.best_no_ask
    if yes_ask is None or no_ask is None:
        return None
    total = yes_ask + no_ask
    if total <= 0:
        return None
    return float(yes_ask / total)


def _market_probability_from_quotes(yes_ask: Optional[Decimal], no_ask: Optional[Decimal]) -> Optional[float]:
    if yes_ask is None or no_ask is None:
        return None
    total = yes_ask + no_ask
    if total <= 0:
        return None
    return float(yes_ask / total)


def _trade_log_path() -> Path:
    return Path(os.getenv("KALSHI_TRADE_LOG", "kalshi_trades.jsonl"))


def _learned_settlements_path() -> Path:
    return Path(os.getenv("KALSHI_LEARNED_SETTLEMENTS", "kalshi_learned_settlements.json"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return asdict(value)
    return str(value)


def log_strategy_record(record: Dict[str, Any]) -> None:
    path = _trade_log_path()
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def _signal_from_record(row: Dict[str, Any]) -> SignalValue:
    return SignalValue(
        name=str(row["name"]),
        value=float(row["value"]),
        confidence=float(row["confidence"]),
        metadata={k: float(v) for k, v in row.get("metadata", {}).items() if isinstance(v, (int, float))},
    )


def _record_timestamp(record: Dict[str, Any]) -> datetime:
    raw = record.get("timestamp")
    if not raw:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _has_live_order(record: Dict[str, Any]) -> bool:
    return bool(record.get("order") and isinstance(record.get("order"), dict) and record["order"].get("order_id"))


def _order_status(record: Dict[str, Any]) -> str:
    order = record.get("order")
    if not isinstance(order, dict):
        return ""
    return str(order.get("status", "")).lower()


def _filled_quantity(record: Dict[str, Any]) -> int:
    order = record.get("order")
    if not isinstance(order, dict):
        return 0
    raw_quantity = (
        order.get("filled_quantity")
        or order.get("fill_count")
        or order.get("fill_count_fp")
        or order.get("count_filled")
        or order.get("remaining_count_fp")
    )
    if raw_quantity in (None, ""):
        return 0
    try:
        return int(Decimal(str(raw_quantity)))
    except Exception:
        return 0


def _has_filled_live_order(record: Dict[str, Any]) -> bool:
    if not _has_live_order(record):
        return False
    status = _order_status(record)
    if status in {"filled", "executed", "partially_filled", "partially-executed"}:
        return True
    return _filled_quantity(record) > 0


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    return Decimal(str(value))


def _load_trade_records() -> List[Dict[str, Any]]:
    path = _trade_log_path()
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSON record in Kalshi trade log")
    return records


def _contract_has_submitted_trade(contract_ticker: str, records: Optional[List[Dict[str, Any]]] = None) -> bool:
    for record in records or _load_trade_records():
        if record.get("contract") != contract_ticker:
            continue
        if _has_live_order(record):
            return True
        trade_intent = record.get("trade_intent") or {}
        if record.get("dry_run") and trade_intent.get("should_trade"):
            return True
    return False


def _build_settlement_update(record: Dict[str, Any], settlement_result: str) -> Optional[SettlementUpdate]:
    if not _has_filled_live_order(record):
        return None

    trade_intent = record.get("trade_intent") or {}
    side = trade_intent.get("side")
    quantity = _filled_quantity(record) or int(trade_intent.get("quantity", 0) or 0)
    entry_price = _to_decimal(trade_intent.get("limit_price"))
    if side not in {"yes", "no"} or quantity <= 0 or entry_price is None:
        return None

    fee = _decimal_env("KALSHI_FEE_PER_CONTRACT", "0.10")
    won = settlement_result == side
    per_contract = (Decimal("1") - entry_price - fee) if won else -(entry_price + fee)
    estimated_pnl = per_contract * Decimal(quantity)
    return SettlementUpdate(
        contract_ticker=str(record.get("contract")),
        result=settlement_result,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        estimated_pnl=estimated_pnl,
        won=won,
    )


async def update_kalshi_weights_from_settlements(
    *,
    kalshi: Optional[KalshiAPI] = None,
    fusion: Optional[KalshiSignalFusion] = None,
) -> List[SettlementUpdate]:
    """
    Reconcile resolved contracts and update adaptive signal weights once.
    """
    trade_log = _trade_log_path()
    if not trade_log.exists():
        return []

    own_kalshi = kalshi is None
    kalshi = kalshi or KalshiAPI()
    fusion = fusion or KalshiSignalFusion(state_path=os.getenv("KALSHI_SIGNAL_STATE", "kalshi_signal_state.json"))
    learned_path = _learned_settlements_path()
    learned = set(json.loads(learned_path.read_text())) if learned_path.exists() else set()

    try:
        try:
            settlements = {item.ticker: item for item in await kalshi.get_settlements()}
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("Kalshi settlement refresh hit rate limit; skipping this cycle")
                return []
            raise
        records_by_contract: Dict[str, List[Dict[str, Any]]] = {}
        for record in _load_trade_records():
            contract = record.get("contract")
            if contract:
                records_by_contract.setdefault(contract, []).append(record)

        updates: List[SettlementUpdate] = []
        for contract, records in records_by_contract.items():
            if contract in learned or contract not in settlements:
                continue
            settlement = settlements[contract]
            if settlement.result not in {"yes", "no"}:
                continue
            records.sort(key=_record_timestamp)
            learning_record = next((record for record in reversed(records) if record.get("signals")), None)
            if not learning_record:
                continue
            signals = [_signal_from_record(row) for row in learning_record.get("signals", [])]
            if not signals:
                continue
            fusion.update_from_outcome(signals, yes_won=settlement.result == "yes")
            trade_record = next((record for record in reversed(records) if _has_filled_live_order(record)), None)
            if trade_record:
                update = _build_settlement_update(trade_record, settlement.result)
                if update:
                    updates.append(update)
            learned.add(contract)

        learned_path.write_text(json.dumps(sorted(learned), indent=2))
        return updates
    finally:
        if own_kalshi:
            await kalshi.close()


async def _current_exposure(kalshi: KalshiAPI, dry_run: bool) -> Optional[Decimal]:
    if dry_run:
        return Decimal("0")
    try:
        positions = await kalshi.get_positions()
        return sum((pos.market_exposure for pos in positions), Decimal("0"))
    except Exception as exc:
        logger.warning(f"Could not fetch Kalshi positions for risk gate: {exc}")
        return None


async def _current_daily_pnl(kalshi: KalshiAPI, dry_run: bool) -> Optional[Decimal]:
    if dry_run:
        return Decimal("0")

    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        settlements = await kalshi.get_settlements(start=start_of_day)
    except Exception as exc:
        logger.warning(f"Could not fetch Kalshi settlements for daily PnL gate: {exc}")
        return None

    settlement_results = {
        settlement.ticker: settlement.result
        for settlement in settlements
        if settlement.result in {"yes", "no"}
    }
    if not settlement_results:
        return Decimal("0")

    realized_pnl = Decimal("0")
    seen_contracts: Set[str] = set()
    records = _load_trade_records()
    for record in reversed(records):
        contract = str(record.get("contract", ""))
        if not contract or contract in seen_contracts or contract not in settlement_results:
            continue
        seen_contracts.add(contract)
        update = _build_settlement_update(record, settlement_results[contract])
        if update and update.estimated_pnl is not None:
            realized_pnl += update.estimated_pnl
    return realized_pnl


def _default_candle_source():
    provider = os.getenv("KALSHI_CANDLE_PROVIDER", "coinbase").lower()
    if provider == "binance":
        return BinanceRESTSource(symbol=os.getenv("KALSHI_SPOT_SYMBOL", "BTCUSDT"))
    return CoinbaseRESTSource(product_id=os.getenv("KALSHI_COINBASE_PRODUCT", "BTC-USD"))


async def run_kalshi_multisignal_strategy(
    *,
    kalshi: Optional[KalshiAPI] = None,
    candle_source: Optional[BinanceRESTSource] = None,
    fusion: Optional[KalshiSignalFusion] = None,
    probability_model: Optional[KalshiProbabilityModel] = None,
    asset: str = "BTC",
    dry_run: Optional[bool] = None,
) -> StrategyRunResult:
    own_kalshi = kalshi is None
    own_candles = candle_source is None
    kalshi = kalshi or KalshiAPI()
    candle_source = candle_source or _default_candle_source()
    fusion = fusion or KalshiSignalFusion(state_path=os.getenv("KALSHI_SIGNAL_STATE", "kalshi_signal_state.json"))
    probability_model = probability_model or KalshiProbabilityModel()
    dry_run = dry_run if dry_run is not None else os.getenv("KALSHI_DRY_RUN", "true").lower() in {"1", "true", "yes"}

    try:
        now = datetime.now(timezone.utc)
        contract = await kalshi.get_active_15m_contract(asset=asset, now=now)
        if not contract:
            return StrategyRunResult(now, None, None, None, None, None, dry_run, "no_active_contract")
        existing_records = _load_trade_records()
        if _contract_has_submitted_trade(contract.ticker, existing_records):
            return StrategyRunResult(now, contract.ticker, None, None, None, None, dry_run, "already_traded_contract")

        open_state = _CONTRACT_OPEN_STATES.get(contract.ticker)
        if open_state is None:
            spread_at_open = None
            if contract.yes_bid is not None and contract.yes_ask is not None:
                spread_at_open = float((contract.yes_ask - contract.yes_bid) * 100)
            elif contract.yes_bid is not None and no_ask is not None:
                derived_yes_ask = Decimal("1") - no_ask
                spread_at_open = float((derived_yes_ask - contract.yes_bid) * 100)
            _CONTRACT_OPEN_STATES[contract.ticker] = ContractOpenState(
                ticker=contract.ticker,
                open_time=now,
                yes_bid=contract.yes_bid,
                no_bid=contract.no_bid,
                spread_at_open=spread_at_open,
            )
            open_state = _CONTRACT_OPEN_STATES[contract.ticker]

        candle_limit = int(os.getenv("KALSHI_CANDLE_LIMIT", "120"))
        candle_rows = await candle_source.get_klines(interval="1m", limit=candle_limit)
        if len(candle_rows) < 35 and own_candles and isinstance(candle_source, BinanceRESTSource):
            fallback = CoinbaseRESTSource(product_id=os.getenv("KALSHI_COINBASE_PRODUCT", "BTC-USD"))
            try:
                logger.info("Falling back to Coinbase candles after Binance returned insufficient data")
                candle_rows = await fallback.get_klines(interval="1m", limit=candle_limit)
            finally:
                await fallback.close()
        candles = _candles_from_rows(candle_rows)
        if len(candles) < 35:
            return StrategyRunResult(now, contract.ticker, None, None, None, None, dry_run, "insufficient_candles")

        orderbook = await kalshi.get_orderbook(contract.ticker, depth=int(os.getenv("KALSHI_ORDERBOOK_DEPTH", "20")))
        indicator_snapshot = compute_indicators(candles)
        signals: List[SignalValue] = normalize_indicators(indicator_snapshot, candles)
        signals.extend(kalshi_market_signals(orderbook))

        fused: FusedKalshiSignal = fusion.fuse(signals)

        tf_disagreement = timeframe_disagreement_signal(candles)
        if tf_disagreement is not None:
            fused = FusedKalshiSignal(
                timestamp=fused.timestamp,
                raw_score=fused.raw_score,
                predicted_prob=fused.predicted_prob,
                confidence=fused.confidence * 0.70,
                signals=fused.signals + [tf_disagreement],
                weights=fused.weights,
            )
            signals = fused.signals

        yes_ask = orderbook.best_yes_ask or contract.yes_ask
        no_ask = orderbook.best_no_ask or contract.no_ask
        market_probability = _market_probability_from_book(orderbook)
        if market_probability is None:
            market_probability = _market_probability_from_quotes(yes_ask, no_ask)

        market_midpoint = float(orderbook.yes_midpoint) if orderbook.yes_midpoint is not None else None

        feature_snapshot = build_feature_snapshot(
            candles=candles,
            orderbook=orderbook,
            contract=contract,
            now=now,
            raw_fusion_probability=fused.predicted_prob,
            spread_at_open=open_state.spread_at_open if open_state else None,
        )
        model_prediction: KalshiModelPrediction = probability_model.predict(
            feature_snapshot,
            fallback_prob=fused.predicted_prob,
            fallback_confidence=fused.confidence,
        )
        risk_config = load_risk_config()
        exposure = await _current_exposure(kalshi, dry_run)
        daily_pnl = await _current_daily_pnl(kalshi, dry_run)

        edge_market_prob = market_midpoint if market_midpoint is not None else market_probability
        minutes_to_expiry = feature_snapshot.features["minutes_to_expiry"]
        intent = build_trade_intent(
            predicted_prob=model_prediction.predicted_prob,
            yes_ask=yes_ask,
            no_ask=no_ask,
            config=risk_config,
            current_exposure=exposure,
            daily_pnl=daily_pnl,
            market_probability=edge_market_prob,
            spread_cents=Decimal(str(feature_snapshot.features["spread_cents"])),
            top_depth=Decimal(str(feature_snapshot.features["top_depth"])),
            model_confidence=model_prediction.confidence,
            seconds_to_expiry=feature_snapshot.features["seconds_to_expiry"],
            minutes_to_expiry=minutes_to_expiry,
        )

        order: Optional[OrderResult] = None
        client_order_id = f"kalshi-{asset.lower()}-{int(now.timestamp())}-{uuid.uuid4().hex[:8]}"
        if intent.should_trade:
            if dry_run:
                logger.info(
                    f"[KALSHI DRY RUN] {contract.ticker} buy {intent.side} "
                    f"{intent.quantity} @ {intent.limit_price_cents}c "
                    f"pred={model_prediction.predicted_prob:.3f} ev={intent.expected_value}"
                )
            else:
                order = await kalshi.place_limit_order(
                    contract_code=contract.ticker,
                    side=intent.side,
                    price_cents=intent.limit_price_cents,
                    quantity=intent.quantity,
                    client_order_id=client_order_id,
                )
                logger.info(f"Placed Kalshi order {order.order_id} for {contract.ticker}")

        log_strategy_record(
            {
                "timestamp": now,
                "asset": asset,
                "contract": contract.ticker,
                "contract_open_time": contract.open_time,
                "contract_close_time": contract.close_time,
                "dry_run": dry_run,
                "predicted_prob": model_prediction.predicted_prob,
                "model_probability": model_prediction.predicted_prob,
                "raw_fusion_probability": fused.predicted_prob,
                "model_confidence": model_prediction.confidence,
                "model_version": model_prediction.model_version,
                "model_fallback_used": model_prediction.fallback_used,
                "market_probability": market_probability,
                "features": feature_snapshot.features,
                "feature_metadata": feature_snapshot.metadata,
                "edge": intent.edge,
                "edge_after_fees": intent.edge_after_fees,
                "trade_allowed": intent.should_trade,
                "block_reason": None if intent.should_trade else intent.reason,
                "sizing_reason": {
                    "kelly_fraction": intent.kelly_fraction,
                    "sizing_multiplier": intent.sizing_multiplier,
                },
                "signals": [asdict(signal) for signal in signals],
                "weights": fused.weights,
                "trade_intent": asdict(intent),
                "client_order_id": client_order_id if intent.should_trade else None,
                "order": asdict(order) if order else None,
            }
        )

        return StrategyRunResult(
            timestamp=now,
            contract_ticker=contract.ticker,
            predicted_prob=model_prediction.predicted_prob,
            market_probability=market_probability,
            trade_intent=intent,
            order=order,
            dry_run=dry_run,
            reason=intent.reason,
        )
    finally:
        if own_kalshi:
            await kalshi.close()
        if own_candles:
            await candle_source.close()

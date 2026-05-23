"""
Event-driven backtest engine for the Kalshi multi-signal strategy.

The default execution path replays the live strategy entrypoint against
simulated market data and uses a single authoritative settlement path.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from core.backtesting.historical_data import BacktestWindow, HistoricalDataLoader
from core.strategy_brain.kalshi_fusion import KalshiSignalFusion
from core.strategy_brain.kalshi_model import KalshiProbabilityModel
from core.strategy_brain.strategies.kalshi_multisignal_strategy import (
    _CONTRACT_OPEN_STATES,
    run_kalshi_multisignal_strategy,
)
from execution.kalshi_api import (
    ContractRef,
    OrderBookLevel,
    OrderBookSnapshot,
    OrderResult,
    PositionSnapshot,
    SettlementResult,
)


@dataclass
class BacktestConfig:
    slippage_cents: float = 0.5
    fee_per_contract: float = 0.01
    synthetic_spread_cents: float = 3.0
    synthetic_top_depth: float = 10.0
    bankroll: float = 1000.0
    kelly_multiplier: float = 0.50
    max_trade_fraction: float = 0.25
    max_trade_dollars: float = 25.0
    max_daily_loss: float = 50.0
    max_total_exposure: float = 100.0
    ev_threshold: float = 0.01
    min_edge_after_fees: float = 0.005
    dry_run: bool = False
    candle_limit: int = 120
    asset: str = "BTC"


@dataclass
class SimulatedOrderBook:
    ticker: str
    yes_bids: List["OrderBookLevel"]
    no_bids: List["OrderBookLevel"]
    timestamp: datetime
    yes_midpoint: Optional[Decimal] = None

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


@dataclass
class SimulatedFill:
    order_id: str
    ticker: str
    side: Literal["yes", "no"]
    filled_price: Decimal
    quantity: int
    slippage_cents: float
    fee: Decimal
    client_order_id: str


@dataclass
class SimulatedPosition:
    ticker: str
    side: Literal["yes", "no"]
    quantity: int
    entry_price: Decimal
    fee_per_contract: Decimal
    current_price: Decimal
    market_exposure: Decimal
    opened_at: datetime


@dataclass
class BacktestTrade:
    window_id: str
    timestamp: datetime
    ticker: str
    side: Literal["yes", "no"]
    quantity: int
    entry_price: Decimal
    fill_price: Decimal
    slippage_cents: float
    fee: Decimal
    market_probability: float
    predicted_probability: float
    model_confidence: float
    intent_edge: Decimal
    signals: List[Dict[str, Any]]
    ground_truth: str
    won: Optional[bool] = None
    settlement_pnl: Optional[Decimal] = None
    settlement_timestamp: Optional[datetime] = None
    signal_names: List[str] = field(default_factory=list)


@dataclass
class ContractResolution:
    ticker: str
    result: str
    settled_time: datetime
    pnl: Decimal


@dataclass
class SettlementResolution:
    def __init__(self) -> None:
        self.resolved: Dict[str, ContractResolution] = {}

    def resolve(self, ticker: str, result: str, settled_time: datetime, pnl: Decimal) -> ContractResolution:
        resolution = ContractResolution(ticker=ticker, result=result, settled_time=settled_time, pnl=pnl)
        self.resolved[ticker] = resolution
        return resolution


@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_edge: float
    avg_edge_winners: float
    avg_edge_losers: float
    total_pnl: Decimal
    max_drawdown: Decimal
    sharpe_annualized: float
    per_signal_accuracy: Dict[str, float]
    per_signal_avg_edge: Dict[str, float]
    per_signal_count: Dict[str, int]
    trade_log: List[Dict[str, Any]]
    regime: str
    data_source: str
    synthetic_used: bool
    regime_assumption: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_edge": self.avg_edge,
            "avg_edge_winners": self.avg_edge_winners,
            "avg_edge_losers": self.avg_edge_losers,
            "total_pnl": str(self.total_pnl),
            "max_drawdown": str(self.max_drawdown),
            "sharpe_annualized": self.sharpe_annualized,
            "per_signal_accuracy": self.per_signal_accuracy,
            "per_signal_avg_edge": self.per_signal_avg_edge,
            "per_signal_count": self.per_signal_count,
            "regime": self.regime,
            "data_source": self.data_source,
            "synthetic_used": self.synthetic_used,
            "regime_assumption": self.regime_assumption,
        }

    def validate_consistency(self) -> None:
        if self.total_trades != self.winning_trades + self.losing_trades:
            raise ValueError("Backtest result is inconsistent: trade counts do not reconcile")
        if self.total_trades > 0 and self.win_rate == 0.0 and self.winning_trades > 0:
            raise ValueError("Backtest result is inconsistent: zero win rate with winning trades recorded")


@dataclass
class SignalAttributor:
    counts: Dict[str, int] = field(default_factory=dict)
    correct: Dict[str, int] = field(default_factory=dict)
    edges: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, signal_name: str, correct: bool, edge: float) -> None:
        self.counts[signal_name] = self.counts.get(signal_name, 0) + 1
        if correct:
            self.correct[signal_name] = self.correct.get(signal_name, 0) + 1
        self.edges.setdefault(signal_name, []).append(edge)

    def summary(self) -> tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        names = set(self.counts)
        accuracy = {
            name: (self.correct.get(name, 0) / self.counts[name]) if self.counts[name] else 0.0
            for name in names
        }
        avg_edge = {
            name: (sum(self.edges.get(name, [])) / len(self.edges.get(name, []))) if self.edges.get(name) else 0.0
            for name in names
        }
        return accuracy, avg_edge, {name: self.counts[name] for name in names}


class PositionTracker:
    def __init__(self) -> None:
        self.positions: Dict[str, SimulatedPosition] = {}
        self.trades: List[BacktestTrade] = []
        self.exposure: Decimal = Decimal("0")
        self.daily_pnl: Decimal = Decimal("0")
        self.total_pnl: Decimal = Decimal("0")
        self.trade_count: int = 0
        self.win_count: int = 0

    def open_position(
        self,
        *,
        window_id: str,
        ticker: str,
        side: Literal["yes", "no"],
        quantity: int,
        entry_price: Decimal,
        fee_per_contract: Decimal,
        slippage_cents: float,
        market_probability: float,
        predicted_probability: float,
        model_confidence: float,
        intent_edge: Decimal,
        signals: List[Dict[str, Any]],
        ground_truth: str,
        timestamp: datetime,
    ) -> BacktestTrade:
        market_exposure = (entry_price + fee_per_contract) * Decimal(quantity)
        trade = BacktestTrade(
            window_id=window_id,
            timestamp=timestamp,
            ticker=ticker,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            fill_price=entry_price,
            slippage_cents=slippage_cents,
            fee=fee_per_contract,
            market_probability=market_probability,
            predicted_probability=predicted_probability,
            model_confidence=model_confidence,
            intent_edge=intent_edge,
            signals=signals,
            ground_truth=ground_truth,
            signal_names=[signal.get("name") for signal in signals],
        )
        self.positions[ticker] = SimulatedPosition(
            ticker=ticker,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            fee_per_contract=fee_per_contract,
            current_price=entry_price,
            market_exposure=market_exposure,
            opened_at=timestamp,
        )
        self.trades.append(trade)
        self.exposure += market_exposure
        self.trade_count += 1
        return trade

    def resolve_position(self, ticker: str, result: str, settled_time: datetime) -> Decimal:
        position = self.positions.get(ticker)
        if position is None:
            return Decimal("0")

        won = result == position.side
        payout = Decimal("1") if won else Decimal("0")
        pnl_per_contract = payout - position.entry_price - position.fee_per_contract
        pnl = pnl_per_contract * Decimal(position.quantity)

        for trade in reversed(self.trades):
            if trade.ticker == ticker:
                trade.won = won
                trade.settlement_pnl = pnl
                trade.settlement_timestamp = settled_time
                break

        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.exposure -= position.market_exposure
        if won:
            self.win_count += 1
        del self.positions[ticker]
        return pnl

    def reset_daily(self) -> None:
        self.daily_pnl = Decimal("0")

    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count else 0.0


class MarketSimulator:
    _BASE_PROBABILITIES = {
        "bull_2024": 0.53,
        "bear_2024": 0.47,
        "sideways_2024_2025": 0.50,
        "bull": 0.53,
        "bear": 0.47,
        "sideways": 0.50,
    }

    def __init__(self, config: BacktestConfig, regime: str = "sideways") -> None:
        self.config = config
        self.regime = regime

    def regime_probability(self) -> float:
        return self._BASE_PROBABILITIES.get(self.regime, 0.50)

    def simulate_orderbook(
        self,
        ticker: str,
        window: BacktestWindow,
        market_probability: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> SimulatedOrderBook:
        midpoint = Decimal(str(market_probability if market_probability is not None else self.regime_probability()))
        half_spread = Decimal(str(self.config.synthetic_spread_cents)) / Decimal("200")
        yes_bid = max(Decimal("0.01"), min(Decimal("0.98"), midpoint - half_spread))
        no_bid = max(Decimal("0.01"), min(Decimal("0.98"), (Decimal("1") - midpoint) - half_spread))
        depth = Decimal(str(self.config.synthetic_top_depth))
        yes_bids = [OrderBookLevel(price=yes_bid - (Decimal("0.001") * i), quantity=depth) for i in range(5)]
        no_bids = [OrderBookLevel(price=no_bid - (Decimal("0.001") * i), quantity=depth) for i in range(5)]
        return SimulatedOrderBook(
            ticker=ticker,
            yes_bids=yes_bids,
            no_bids=no_bids,
            timestamp=timestamp or window.open_time,
            yes_midpoint=midpoint,
        )


class TradeSimulation:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def simulate_fill(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        limit_price_cents: int,
        quantity: int,
        client_order_id: str,
        ts: datetime,
    ) -> SimulatedFill:
        price = Decimal(str(limit_price_cents)) / Decimal("100")
        slippage = Decimal(str(self.config.slippage_cents)) / Decimal("100")
        fill_price = min(Decimal("0.99"), price + slippage)
        return SimulatedFill(
            order_id=f"sim_{uuid.uuid4().hex[:12]}",
            ticker=ticker,
            side=side,
            filled_price=fill_price,
            quantity=quantity,
            slippage_cents=self.config.slippage_cents,
            fee=Decimal(str(self.config.fee_per_contract)),
            client_order_id=client_order_id,
        )


class SimulatedCandleSource:
    def __init__(self, candles: Iterable[Any], *, limit: int = 120) -> None:
        self.candles = sorted(candles, key=lambda candle: candle.timestamp)
        self.limit = limit
        self.current_time: Optional[datetime] = None

    def set_time(self, current_time: datetime) -> None:
        self.current_time = current_time

    async def get_klines(self, interval: str = "1m", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        current = self.current_time or datetime.now(timezone.utc)
        rows = [candle for candle in self.candles if candle.timestamp < current]
        rows = rows[-(limit or self.limit) :]
        return [
            {
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "timestamp": candle.timestamp,
            }
            for candle in rows
        ]

    async def close(self) -> None:
        return None


class SimulatedKalshiAPI:
    def __init__(self, market_sim: MarketSimulator, trade_sim: TradeSimulation) -> None:
        self.market_sim = market_sim
        self.trade_sim = trade_sim
        self.current_window: Optional[BacktestWindow] = None
        self.current_contract: Optional[ContractRef] = None
        self.current_orderbook: Optional[OrderBookSnapshot] = None
        self.fills: Dict[str, SimulatedFill] = {}
        self.settlement_history: List[SettlementResult] = []

    def set_window(self, window: BacktestWindow) -> None:
        self.current_window = window
        ticker = f"SIM_{window.window_id}"
        book = self.market_sim.simulate_orderbook(ticker, window)
        self.current_orderbook = OrderBookSnapshot(
            ticker=ticker,
            yes_bids=[OrderBookLevel(price=level.price, quantity=level.quantity) for level in book.yes_bids],
            no_bids=[OrderBookLevel(price=level.price, quantity=level.quantity) for level in book.no_bids],
            timestamp=book.timestamp,
        )
        self.current_contract = ContractRef(
            ticker=ticker,
            event_ticker=None,
            title=f"Simulated {ticker}",
            open_time=window.open_time,
            close_time=window.close_time,
            status="open",
            yes_bid=self.current_orderbook.best_yes_bid,
            yes_ask=self.current_orderbook.best_yes_ask,
            no_bid=self.current_orderbook.best_no_bid,
            no_ask=self.current_orderbook.best_no_ask,
            open_spread_cents=float(
                ((self.current_orderbook.best_yes_ask or Decimal("0")) - (self.current_orderbook.best_yes_bid or Decimal("0"))) * 100
            ),
        )

    async def get_active_15m_contract(
        self,
        asset: str = "BTC",
        now: Optional[datetime] = None,
    ) -> Optional[ContractRef]:
        return self.current_contract

    async def get_orderbook(self, contract_ticker: str, depth: int = 20) -> OrderBookSnapshot:
        if self.current_orderbook is None or self.current_orderbook.ticker != contract_ticker:
            raise ValueError(f"Unknown contract {contract_ticker}")
        return self.current_orderbook

    async def place_limit_order(
        self,
        contract_code: str,
        side: Literal["yes", "no"],
        price_cents: int,
        quantity: int,
        client_order_id: str,
    ) -> OrderResult:
        fill = self.trade_sim.simulate_fill(
            contract_code,
            side,
            price_cents,
            quantity,
            client_order_id,
            self.current_window.open_time if self.current_window else datetime.now(timezone.utc),
        )
        self.fills[contract_code] = fill
        return OrderResult(
            order_id=fill.order_id,
            client_order_id=fill.client_order_id,
            ticker=fill.ticker,
            side=fill.side,
            status="filled",
            filled_qty=fill.quantity,
            remaining_qty=Decimal("0"),
            raw={"filled_quantity": str(fill.quantity)},
        )

    async def get_positions(self) -> List[PositionSnapshot]:
        return []

    async def get_settlements(self, start: Optional[datetime] = None) -> List[SettlementResult]:
        if start is None:
            return list(self.settlement_history)
        return [settlement for settlement in self.settlement_history if settlement.settled_time and settlement.settled_time >= start]

    def record_settlement(self, ticker: str, result: str, settled_time: datetime) -> None:
        self.settlement_history.append(
            SettlementResult(
                ticker=ticker,
                result=result,
                settled_time=settled_time,
                raw={},
            )
        )

    def get_fill(self, ticker: str) -> Optional[SimulatedFill]:
        return self.fills.get(ticker)

    async def close(self) -> None:
        return None


@contextmanager
def _temporary_strategy_env(temp_dir: Path, config: BacktestConfig):
    env_updates = {
        "KALSHI_TRADE_LOG": str(temp_dir / "kalshi_trades.jsonl"),
        "KALSHI_LEARNED_SETTLEMENTS": str(temp_dir / "kalshi_learned_settlements.json"),
        "KALSHI_SIGNAL_STATE": str(temp_dir / "kalshi_signal_state.json"),
        "KALSHI_CANDLE_LIMIT": str(config.candle_limit),
        "KALSHI_BANKROLL": str(config.bankroll),
        "KALSHI_KELLY_CAP": str(config.max_trade_fraction),
        "KALSHI_KELLY_MULTIPLIER": str(config.kelly_multiplier),
        "KALSHI_MAX_TRADE_DOLLARS": str(config.max_trade_dollars),
        "KALSHI_MAX_DAILY_LOSS": str(config.max_daily_loss),
        "KALSHI_MAX_TOTAL_EXPOSURE": str(config.max_total_exposure),
        "KALSHI_EV_THRESHOLD": str(config.ev_threshold),
        "KALSHI_FEE_PER_CONTRACT": str(config.fee_per_contract),
        "KALSHI_MIN_EDGE_AFTER_FEES": str(config.min_edge_after_fees),
        "KALSHI_ENABLE_DERIBIT_SIGNALS": "false",
        "KALSHI_ENABLE_SENTIMENT_SIGNAL": "false",
    }
    previous = {key: os.environ.get(key) for key in env_updates}
    try:
        for key, value in env_updates.items():
            os.environ[key] = value
        yield env_updates
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


class KalshiBacktester:
    def __init__(
        self,
        windows: List[BacktestWindow],
        config: Optional[BacktestConfig] = None,
        regime: str = "sideways",
        strategy_fn: Optional[Callable[..., Any]] = None,
        *,
        data_source: str = "unknown",
        synthetic_used: bool = False,
        regime_assumption: Optional[str] = None,
    ) -> None:
        self.windows = windows
        self.config = config or BacktestConfig()
        self.regime = regime
        self.strategy_fn = strategy_fn
        self.data_source = data_source
        self.synthetic_used = synthetic_used
        self.regime_assumption = regime_assumption or regime

        self.market_sim = MarketSimulator(self.config, regime=regime)
        self.trade_sim = TradeSimulation(self.config)
        self.api = SimulatedKalshiAPI(self.market_sim, self.trade_sim)
        self.tracker = PositionTracker()
        self.attributor = SignalAttributor()
        self.settlements = SettlementResolution()
        self.candle_source = SimulatedCandleSource(self._flatten_candles(), limit=self.config.candle_limit)

    def _flatten_candles(self) -> List[Any]:
        candles: List[Any] = []
        seen: set[datetime] = set()
        for window in self.windows:
            for candle in window.candles:
                if candle.timestamp in seen:
                    continue
                seen.add(candle.timestamp)
                candles.append(candle)
        return candles

    def _compute_max_drawdown(self, equity_curve: List[float]) -> float:
        peak = equity_curve[0]
        max_drawdown = 0.0
        for value in equity_curve:
            peak = max(peak, value)
            max_drawdown = max(max_drawdown, peak - value)
        return max_drawdown

    def _compute_sharpe(self, equity_curve: List[float], periods_per_year: int = 96) -> float:
        if len(equity_curve) < 3:
            return 0.0
        returns: List[float] = []
        for previous, current in zip(equity_curve[:-1], equity_curve[1:]):
            if previous <= 0:
                continue
            returns.append((current - previous) / previous)
        if len(returns) < 2:
            return 0.0
        avg = sum(returns) / len(returns)
        variance = sum((ret - avg) ** 2 for ret in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        return (avg / std) * math.sqrt(periods_per_year)

    def _record_signal_outcome(self, trade: BacktestTrade, realized_pnl: Decimal) -> None:
        for signal in trade.signals:
            value = float(signal.get("value", 0.0) or 0.0)
            if value == 0.0:
                continue
            correct = (value > 0 and trade.ground_truth == "yes") or (value < 0 and trade.ground_truth == "no")
            self.attributor.record(str(signal.get("name")), correct, float(realized_pnl))

    async def _run_strategy_window(
        self,
        window: BacktestWindow,
        *,
        trade_log_path: Path,
        fusion: KalshiSignalFusion,
        probability_model: KalshiProbabilityModel,
    ) -> Optional[BacktestTrade]:
        self.api.set_window(window)
        self.candle_source.set_time(window.open_time)
        _CONTRACT_OPEN_STATES.clear()
        before = len(_read_records(trade_log_path))

        if self.strategy_fn is not None:
            return await self.strategy_fn(window, self.api, self.market_sim, self.tracker)

        await run_kalshi_multisignal_strategy(
            kalshi=self.api,
            candle_source=self.candle_source,
            fusion=fusion,
            probability_model=probability_model,
            asset=self.config.asset,
            dry_run=False,
            now=window.open_time,
        )

        records = _read_records(trade_log_path)
        if len(records) <= before:
            return None
        record = records[-1]
        order = record.get("order") or {}
        trade_intent = record.get("trade_intent") or {}
        if not record.get("trade_allowed") or not order:
            return None
        fill = self.api.get_fill(str(record["contract"]))
        if fill is None:
            return None
        return self.tracker.open_position(
            window_id=window.window_id,
            ticker=str(record["contract"]),
            side=str(trade_intent["side"]),
            quantity=int(trade_intent["quantity"]),
            entry_price=fill.filled_price,
            fee_per_contract=fill.fee,
            slippage_cents=fill.slippage_cents,
            market_probability=float(record.get("market_probability") or 0.5),
            predicted_probability=float(record.get("predicted_prob") or 0.5),
            model_confidence=float(record.get("model_confidence") or 0.0),
            intent_edge=Decimal(str(record.get("edge_after_fees", record.get("edge", 0.0)) or 0.0)),
            signals=list(record.get("signals") or []),
            ground_truth=window.ground_truth,
            timestamp=window.open_time,
        )

    async def run(self) -> BacktestResult:
        equity_curve = [float(self.config.bankroll)]
        current_day = None

        with tempfile.TemporaryDirectory(prefix="kalshi_backtest_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            trade_log_path = temp_dir / "kalshi_trades.jsonl"
            missing_model_path = temp_dir / "missing_model.json"
            with _temporary_strategy_env(temp_dir, self.config):
                fusion = KalshiSignalFusion(state_path=str(temp_dir / "kalshi_signal_state.json"))
                probability_model = KalshiProbabilityModel(model_path=str(missing_model_path))

                for window in self.windows:
                    if current_day != window.open_time.date():
                        self.tracker.reset_daily()
                        current_day = window.open_time.date()

                    trade = await self._run_strategy_window(
                        window,
                        trade_log_path=trade_log_path,
                        fusion=fusion,
                        probability_model=probability_model,
                    )

                    if trade is not None:
                        pnl = self.tracker.resolve_position(trade.ticker, window.ground_truth, window.close_time)
                        self.settlements.resolve(trade.ticker, window.ground_truth, window.close_time, pnl)
                        self.api.record_settlement(trade.ticker, window.ground_truth, window.close_time)
                        self._record_signal_outcome(trade, pnl)

                    equity_curve.append(float(self.config.bankroll) + float(self.tracker.total_pnl))

        trades = self.tracker.trades
        winning = [trade for trade in trades if trade.won]
        losing = [trade for trade in trades if trade.won is False]
        per_signal_accuracy, per_signal_avg_edge, per_signal_count = self.attributor.summary()

        result = BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=self.tracker.win_rate,
            avg_edge=sum(float(trade.intent_edge) for trade in trades) / len(trades) if trades else 0.0,
            avg_edge_winners=sum(float(trade.intent_edge) for trade in winning) / len(winning) if winning else 0.0,
            avg_edge_losers=sum(float(trade.intent_edge) for trade in losing) / len(losing) if losing else 0.0,
            total_pnl=self.tracker.total_pnl,
            max_drawdown=Decimal(str(self._compute_max_drawdown(equity_curve))),
            sharpe_annualized=self._compute_sharpe(equity_curve),
            per_signal_accuracy=per_signal_accuracy,
            per_signal_avg_edge=per_signal_avg_edge,
            per_signal_count=per_signal_count,
            trade_log=[json.loads(json.dumps(asdict(trade), default=_json_default)) for trade in trades],
            regime=self.regime,
            data_source=self.data_source,
            synthetic_used=self.synthetic_used,
            regime_assumption=self.regime_assumption,
        )
        result.validate_consistency()
        return result


async def run_regime_backtest(
    regime_label: str,
    start: datetime,
    end: datetime,
    config: Optional[BacktestConfig] = None,
    strategy_fn: Optional[Callable[..., Any]] = None,
) -> BacktestResult:
    loader = HistoricalDataLoader(regime=regime_label)
    windows = await loader.load_windows(start, end)
    await loader.close()
    if not windows:
        raise ValueError(f"No backtest windows loaded for regime '{regime_label}' [{start} – {end}]")
    tester = KalshiBacktester(
        windows,
        config=config,
        regime=regime_label,
        strategy_fn=strategy_fn,
        data_source=loader.last_data_source,
        synthetic_used=loader.last_synthetic_used,
        regime_assumption=loader.regime,
    )
    return await tester.run()

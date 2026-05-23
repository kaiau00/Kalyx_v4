"""
Performance tracking and signal attribution for the trading bot.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class Trade:
    trade_id: str
    timestamp: datetime
    direction: str
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    pnl: Decimal
    pnl_pct: float
    duration_seconds: float
    signal_score: float
    signal_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalAttribution:
    edge_total: float = 0.0
    correct_count: int = 0
    count: int = 0
    latest_weight: float = 0.0

    @property
    def avg_edge(self) -> float:
        return self.edge_total / self.count if self.count else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.count if self.count else 0.0


@dataclass
class PerformanceMetrics:
    timestamp: datetime
    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    open_positions: int
    avg_position_size: Decimal
    avg_hold_time: float
    total_exposure: Decimal
    risk_utilization: float
    avg_signal_score: float
    avg_signal_confidence: float


class PerformanceTracker:
    def __init__(self, initial_capital: Decimal = Decimal("1000.0")):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self._trades: List[Trade] = []
        self._metrics_history: deque = deque(maxlen=10000)
        self._last_metrics: Optional[PerformanceMetrics] = None
        self._metrics_dirty = True
        self._peak_capital = initial_capital
        self._signal_attribution: Dict[str, SignalAttribution] = {}
        self._settlement_latencies: deque = deque(maxlen=5000)
        self._edge_distribution: deque = deque(maxlen=5000)
        self._signal_weight_history: List[Dict[str, Any]] = []
        logger.info(f"Initialized Performance Tracker (capital=${initial_capital})")

    def record_trade(
        self,
        trade_id: str,
        direction: str,
        entry_price: Decimal,
        exit_price: Decimal,
        size: Decimal,
        entry_time: datetime,
        exit_time: datetime,
        signal_score: float = 0.0,
        signal_confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Trade:
        pnl_pct = (exit_price - entry_price) / entry_price if direction == "long" else (entry_price - exit_price) / entry_price
        pnl = size * pnl_pct
        trade = Trade(
            trade_id=trade_id,
            timestamp=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            pnl_pct=float(pnl_pct),
            duration_seconds=(exit_time - entry_time).total_seconds(),
            signal_score=signal_score,
            signal_confidence=signal_confidence,
            metadata=metadata or {},
        )
        self._trades.append(trade)
        self.current_capital += pnl
        self._peak_capital = max(self._peak_capital, self.current_capital)
        self._metrics_dirty = True
        return trade

    def record_signal_attribution(self, signal_name: str, edge: float, correct: bool, weight: float = 0.0) -> None:
        attribution = self._signal_attribution.setdefault(signal_name, SignalAttribution())
        attribution.edge_total += float(edge)
        attribution.count += 1
        attribution.correct_count += int(correct)
        attribution.latest_weight = float(weight)

    def record_signal_weights(self, weights: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        point = {"timestamp": (timestamp or datetime.utcnow()).isoformat(), "weights": dict(weights)}
        self._signal_weight_history.append(point)
        if len(self._signal_weight_history) > 10000:
            self._signal_weight_history = self._signal_weight_history[-10000:]

    def record_settlement_latency(self, close_time: datetime, settled_time: datetime) -> None:
        self._settlement_latencies.append(max(0.0, (settled_time - close_time).total_seconds()))

    def record_edge(self, edge_after_fees: float) -> None:
        self._edge_distribution.append(float(edge_after_fees))

    def get_signal_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "count": float(stats.count),
                "avg_edge": stats.avg_edge,
                "accuracy": stats.accuracy,
                "weight": stats.latest_weight,
            }
            for name, stats in self._signal_attribution.items()
        }

    def get_edge_distribution(self) -> List[float]:
        return list(self._edge_distribution)

    def get_avg_settlement_latency(self) -> float:
        if not self._settlement_latencies:
            return 0.0
        return sum(self._settlement_latencies) / len(self._settlement_latencies)

    def get_signal_weight_history(self) -> List[Dict[str, Any]]:
        return list(self._signal_weight_history)

    def calculate_metrics(self, force: bool = False) -> PerformanceMetrics:
        if not force and not self._metrics_dirty and self._last_metrics:
            return self._last_metrics

        total_pnl = self.current_capital - self.initial_capital
        total_trades = len(self._trades)
        winning_trades = len([t for t in self._trades if t.pnl > 0])
        losing_trades = len([t for t in self._trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades else 0.0
        roi = float(total_pnl / self.initial_capital) if self.initial_capital else 0.0
        sharpe = self._calculate_sharpe_ratio()
        max_drawdown = float((self._peak_capital - self.current_capital) / self._peak_capital) if self._peak_capital else 0.0

        avg_size = sum((t.size for t in self._trades), Decimal("0")) / total_trades if total_trades else Decimal("0")
        avg_hold = sum(t.duration_seconds for t in self._trades) / total_trades if total_trades else 0.0
        avg_score = sum(t.signal_score for t in self._trades) / total_trades if total_trades else 0.0
        avg_conf = sum(t.signal_confidence for t in self._trades) / total_trades if total_trades else 0.0

        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            total_pnl=total_pnl,
            realized_pnl=total_pnl,
            unrealized_pnl=Decimal("0"),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            roi=roi,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            open_positions=0,
            avg_position_size=avg_size,
            avg_hold_time=avg_hold,
            total_exposure=Decimal("0"),
            risk_utilization=0.0,
            avg_signal_score=avg_score,
            avg_signal_confidence=avg_conf,
        )
        self._last_metrics = metrics
        self._metrics_dirty = False
        self._metrics_history.append(metrics)
        return metrics

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if len(self._trades) < 2:
            return 0.0
        returns = [float(t.pnl / t.size) for t in self._trades if t.size > 0]
        if not returns:
            return 0.0
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = variance ** 0.5
        if std_return == 0:
            return 0.0
        return (mean_return - risk_free_rate / 252) / std_return * (252 ** 0.5)

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        curve = [{"timestamp": (self._trades[0].timestamp if self._trades else datetime.now()), "equity": float(self.initial_capital)}]
        running = self.initial_capital
        for trade in self._trades:
            running += trade.pnl
            curve.append({"timestamp": trade.timestamp, "equity": float(running)})
        return curve

    def get_daily_pnl(self, days: int = 30) -> List[Dict[str, Any]]:
        cutoff = datetime.now() - timedelta(days=days)
        daily: Dict[str, Decimal] = {}
        for trade in self._trades:
            if trade.timestamp < cutoff:
                continue
            key = trade.timestamp.strftime("%Y-%m-%d")
            daily[key] = daily.get(key, Decimal("0")) + trade.pnl
        return [{"date": day, "pnl": float(pnl)} for day, pnl in sorted(daily.items())]

    def export_for_grafana(self) -> Dict[str, Any]:
        metrics = self.calculate_metrics()
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_pnl": float(metrics.total_pnl),
                "roi": metrics.roi * 100,
                "win_rate": metrics.win_rate * 100,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown * 100,
                "total_trades": metrics.total_trades,
                "current_capital": float(self.current_capital),
                "settlement_latency_seconds": self.get_avg_settlement_latency(),
            },
            "equity_curve": self.get_equity_curve(),
            "daily_pnl": self.get_daily_pnl(30),
            "signal_summary": self.get_signal_summary(),
            "signal_weight_history": self.get_signal_weight_history(),
            "edge_distribution": self.get_edge_distribution(),
        }


_performance_tracker_instance = None


def get_performance_tracker() -> PerformanceTracker:
    global _performance_tracker_instance
    if _performance_tracker_instance is None:
        _performance_tracker_instance = PerformanceTracker()
    return _performance_tracker_instance

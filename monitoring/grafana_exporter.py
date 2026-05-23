"""
Prometheus exporter for trading and attribution metrics.
"""
from __future__ import annotations

import asyncio
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, REGISTRY, generate_latest

from execution.execution_engine import get_execution_engine
from execution.risk_engine import get_risk_engine
from monitoring.performance_tracker import get_performance_tracker


class MetricsHandler(BaseHTTPRequestHandler):
    exporter = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/", ""}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Kalyx Metrics</h1><a href='/metrics'>/metrics</a></body></html>")
            return
        if parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"healthy"}')
            return
        if parsed.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(generate_latest(REGISTRY))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path == "/metrics":
            return self.do_GET()
        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def log_message(self, format, *args):
        return


class GrafanaMetricsExporter:
    def __init__(self, port: int = 8000, update_interval: int = 5):
        self.port = port
        self.update_interval = update_interval
        self.performance = get_performance_tracker()
        self.risk = get_risk_engine()
        self.execution = get_execution_engine()
        self._is_running = False
        self._server = None
        self._thread = None
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        self.total_pnl = Gauge("trading_total_pnl", "Total profit/loss in USD")
        self.roi = Gauge("trading_roi", "Return on investment as percentage")
        self.win_rate = Gauge("trading_win_rate", "Percentage of winning trades")
        self.sharpe_ratio = Gauge("trading_sharpe_ratio", "Sharpe ratio")
        self.max_drawdown = Gauge("trading_max_drawdown", "Maximum drawdown as percentage")
        self.open_positions = Gauge("trading_open_positions", "Number of open positions")
        self.total_exposure = Gauge("trading_total_exposure", "Total exposure in USD")
        self.risk_utilization = Gauge("trading_risk_utilization", "Risk utilization percentage")
        self.current_capital = Gauge("trading_current_capital", "Current account capital in USD")
        self.avg_signal_score = Gauge("trading_avg_signal_score", "Average signal score")
        self.avg_signal_confidence = Gauge("trading_avg_signal_confidence", "Average signal confidence")
        self.cumulative_pnl = Gauge("trading_total_pnl_cumulative", "Cumulative PnL")
        self.drawdown = Gauge("trading_drawdown", "Current drawdown percentage")
        self.settlement_latency = Gauge("trading_settlement_latency_seconds", "Average settlement latency in seconds")
        self.total_trades = Counter("trading_total_trades_total", "Total number of trades executed")
        self.winning_trades = Counter("trading_winning_trades_total", "Number of winning trades")
        self.losing_trades = Counter("trading_losing_trades_total", "Number of losing trades")
        self.orders_placed = Counter("trading_orders_placed_total", "Orders placed")
        self.orders_filled = Counter("trading_orders_filled_total", "Orders filled")
        self.orders_rejected = Counter("trading_orders_rejected_total", "Orders rejected")
        self.trade_duration = Histogram("trading_trade_duration_seconds", "Trade duration", buckets=[60, 300, 900, 1800, 3600, 7200, 14400])
        self.edge_distribution = Histogram("trading_edge_after_fees", "Edge after fees", buckets=[-0.20, -0.10, -0.05, 0.0, 0.03, 0.05, 0.08, 0.12, 0.20])
        self.signal_edge = Gauge("signal_edge_avg", "Average edge per signal", ["signal_name"])
        self.signal_accuracy = Gauge("signal_accuracy_pct", "Accuracy percentage per signal", ["signal_name"])
        self.signal_count = Gauge("signal_count_total", "Trade count per signal", ["signal_name"])
        self.signal_weight = Gauge("signal_weight", "Current weight per signal", ["signal_name"])

    def update_metrics(self) -> None:
        perf_metrics = self.performance.calculate_metrics()
        self.total_pnl.set(float(perf_metrics.total_pnl))
        self.roi.set(perf_metrics.roi * 100)
        self.win_rate.set(perf_metrics.win_rate * 100)
        self.sharpe_ratio.set(perf_metrics.sharpe_ratio)
        self.max_drawdown.set(perf_metrics.max_drawdown * 100)
        self.open_positions.set(perf_metrics.open_positions)
        self.total_exposure.set(float(perf_metrics.total_exposure))
        self.avg_signal_score.set(perf_metrics.avg_signal_score)
        self.avg_signal_confidence.set(perf_metrics.avg_signal_confidence)
        self.current_capital.set(float(self.performance.current_capital))
        self.cumulative_pnl.set(float(perf_metrics.total_pnl))
        self.drawdown.set(perf_metrics.max_drawdown * 100)
        self.settlement_latency.set(self.performance.get_avg_settlement_latency())

        risk_summary = self.risk.get_risk_summary()
        if risk_summary:
            self.risk_utilization.set(risk_summary["exposure"]["utilization_pct"])

        signal_summary = self.performance.get_signal_summary()
        for name, values in signal_summary.items():
            self.signal_edge.labels(signal_name=name).set(values["avg_edge"])
            self.signal_accuracy.labels(signal_name=name).set(values["accuracy"] * 100)
            self.signal_count.labels(signal_name=name).set(values["count"])
            self.signal_weight.labels(signal_name=name).set(values["weight"])

    async def start(self) -> None:
        if self._is_running:
            return
        MetricsHandler.exporter = self
        self._server = HTTPServer(("0.0.0.0", self.port), MetricsHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._is_running = True
        asyncio.create_task(self._update_loop())

    async def _update_loop(self) -> None:
        while self._is_running:
            try:
                self.update_metrics()
            except Exception as exc:
                logger.error(f"Error updating metrics: {exc}")
            await asyncio.sleep(self.update_interval)

    async def stop(self) -> None:
        self._is_running = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def increment_trade_counter(self, won: bool) -> None:
        self.total_trades.inc()
        if won:
            self.winning_trades.inc()
        else:
            self.losing_trades.inc()

    def record_trade_duration(self, duration_seconds: float) -> None:
        self.trade_duration.observe(duration_seconds)

    def increment_order_counter(self, status: str) -> None:
        if status == "placed":
            self.orders_placed.inc()
        elif status == "filled":
            self.orders_filled.inc()
        elif status == "rejected":
            self.orders_rejected.inc()

    def record_signal_attribution(self, signal_name: str, edge: float, correct: bool, weight: float = 0.0) -> None:
        self.performance.record_signal_attribution(signal_name, edge, correct, weight)
        values = self.performance.get_signal_summary().get(signal_name, {})
        self.signal_edge.labels(signal_name=signal_name).set(values.get("avg_edge", 0.0))
        self.signal_accuracy.labels(signal_name=signal_name).set(values.get("accuracy", 0.0) * 100)
        self.signal_count.labels(signal_name=signal_name).set(values.get("count", 0.0))
        self.signal_weight.labels(signal_name=signal_name).set(values.get("weight", 0.0))

    def record_signal_weights(self, weights: dict[str, float]) -> None:
        self.performance.record_signal_weights(weights)
        for name, value in weights.items():
            self.signal_weight.labels(signal_name=name).set(value)

    def record_edge(self, edge_after_fees: float) -> None:
        self.performance.record_edge(edge_after_fees)
        self.edge_distribution.observe(edge_after_fees)

    def record_settlement_latency(self, close_time, settled_time) -> None:
        self.performance.record_settlement_latency(close_time, settled_time)
        self.settlement_latency.set(self.performance.get_avg_settlement_latency())


_grafana_exporter_instance = None


def get_grafana_exporter() -> GrafanaMetricsExporter:
    global _grafana_exporter_instance
    if _grafana_exporter_instance is None:
        _grafana_exporter_instance = GrafanaMetricsExporter()
    return _grafana_exporter_instance

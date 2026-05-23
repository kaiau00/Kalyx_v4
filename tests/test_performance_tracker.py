from datetime import datetime, timedelta
from decimal import Decimal

from monitoring.performance_tracker import PerformanceTracker


def test_signal_summary_tracks_edge_accuracy_and_weight():
    tracker = PerformanceTracker(initial_capital=Decimal("1000"))
    tracker.record_signal_attribution("rsi", edge=0.05, correct=True, weight=0.2)
    tracker.record_signal_attribution("rsi", edge=-0.01, correct=False, weight=0.25)
    summary = tracker.get_signal_summary()

    assert summary["rsi"]["count"] == 2.0
    assert summary["rsi"]["avg_edge"] == 0.02
    assert summary["rsi"]["accuracy"] == 0.5
    assert summary["rsi"]["weight"] == 0.25


def test_settlement_latency_and_edge_distribution_are_recorded():
    tracker = PerformanceTracker(initial_capital=Decimal("1000"))
    now = datetime.utcnow()
    tracker.record_settlement_latency(now, now + timedelta(seconds=90))
    tracker.record_edge(0.04)

    assert tracker.get_avg_settlement_latency() == 90.0
    assert tracker.get_edge_distribution() == [0.04]

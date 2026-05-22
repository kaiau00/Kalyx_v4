"""
Kalshi multi-signal fusion and adaptive signal weight tracking.
"""
import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from core.strategy_brain.kalshi_indicators import SignalValue


DEFAULT_SIGNAL_NAMES = [
    "rsi",
    "macd",
    "cci",
    "fisher",
    "adx",
    "atr_regime",
    "book_imbalance_d3",
    "book_imbalance_d5",
    "book_imbalance_d10",
    "book_imbalance_d20",
    "market_probability",
    "crowd_fade",
    "timeframe_disagreement",
]


@dataclass
class FusedKalshiSignal:
    timestamp: datetime
    raw_score: float
    predicted_prob: float
    confidence: float
    signals: List[SignalValue]
    weights: Dict[str, float]

    @property
    def direction(self) -> str:
        return "yes" if self.predicted_prob >= 0.5 else "no"


@dataclass
class SignalStats:
    recent_correct: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def accuracy(self) -> float:
        if not self.recent_correct:
            return 0.5
        return sum(1 for c in self.recent_correct if c) / len(self.recent_correct)


@dataclass
class FusionState:
    weights: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, SignalStats] = field(default_factory=dict)


class KalshiSignalFusion:
    def __init__(
        self,
        signal_names: Optional[Iterable[str]] = None,
        state_path: str = "kalshi_signal_state.json",
        learning_rate: float = 0.10,
        min_samples: int = 5,
    ) -> None:
        self.signal_names = list(signal_names or DEFAULT_SIGNAL_NAMES)
        self.state_path = Path(state_path)
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.state = self._load_state()
        self._ensure_defaults()

    def _load_state(self) -> FusionState:
        if not self.state_path.exists():
            return FusionState()
        try:
            data = json.loads(self.state_path.read_text())
            stats = {}
            for k, v in data.get("stats", {}).items():
                recent = deque(v.get("recent_correct", []), maxlen=50)
                stats[k] = SignalStats(recent_correct=recent)
            return FusionState(
                weights={k: float(v) for k, v in data.get("weights", {}).items()},
                stats=stats,
            )
        except Exception:
            return FusionState()

    def _ensure_defaults(self) -> None:
        if not self.state.weights:
            equal = 1.0 / len(self.signal_names)
            self.state.weights = {name: equal for name in self.signal_names}
        for name in self.signal_names:
            self.state.weights.setdefault(name, 0.0)
            self.state.stats.setdefault(name, SignalStats())
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(max(0.0, weight) for weight in self.state.weights.values())
        if total <= 0:
            equal = 1.0 / len(self.signal_names)
            self.state.weights = {name: equal for name in self.signal_names}
            return
        self.state.weights = {
            name: max(0.0, self.state.weights.get(name, 0.0)) / total
            for name in self.signal_names
        }

    def save(self) -> None:
        payload = {
            "weights": self.state.weights,
            "stats": {
                name: {"recent_correct": list(stats.recent_correct)}
                for name, stats in self.state.stats.items()
            },
        }
        self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def fuse(self, signals: List[SignalValue]) -> FusedKalshiSignal:
        if not signals:
            return FusedKalshiSignal(
                timestamp=datetime.now(timezone.utc),
                raw_score=0.0,
                predicted_prob=0.5,
                confidence=0.0,
                signals=[],
                weights=self.state.weights.copy(),
            )

        raw_score = 0.0
        used_weight = 0.0
        confidence_parts = []
        for signal in signals:
            weight = self.state.weights.get(signal.name, 0.0)
            raw_score += weight * signal.value * max(0.0, min(1.0, signal.confidence))
            used_weight += weight
            confidence_parts.append(abs(signal.value) * signal.confidence)

        normalized_score = raw_score / used_weight if used_weight > 0 else 0.0
        predicted_prob = 1.0 / (1.0 + math.exp(-3.0 * normalized_score))
        confidence = sum(confidence_parts) / len(confidence_parts) if confidence_parts else 0.0
        return FusedKalshiSignal(
            timestamp=datetime.now(timezone.utc),
            raw_score=normalized_score,
            predicted_prob=predicted_prob,
            confidence=max(0.0, min(1.0, confidence)),
            signals=signals,
            weights=self.state.weights.copy(),
        )

    def update_from_outcome(self, signals: List[SignalValue], yes_won: bool) -> Dict[str, float]:
        actual_direction = 1.0 if yes_won else -1.0
        for signal in signals:
            stats = self.state.stats.setdefault(signal.name, SignalStats())
            if signal.value == 0:
                continue
            is_correct = bool(signal.value * actual_direction > 0)
            stats.recent_correct.append(is_correct)

        for name, stats in self.state.stats.items():
            if len(stats.recent_correct) < self.min_samples:
                continue
            target = max(0.03, min(0.35, stats.accuracy))
            current = self.state.weights.get(name, 0.0)
            self.state.weights[name] = current + (target - current) * self.learning_rate

        self._normalize_weights()
        self.save()
        return self.state.weights.copy()

from decimal import Decimal

from core.strategy_brain.kalshi_fusion import KalshiSignalFusion
from core.strategy_brain.kalshi_indicators import SignalValue
from execution.kalshi_risk import KalshiRiskConfig, build_trade_intent, expected_value


def test_fusion_defaults_to_equal_weights(tmp_path):
    fusion = KalshiSignalFusion(signal_names=["a", "b"], state_path=str(tmp_path / "state.json"))

    assert fusion.state.weights == {"a": 0.5, "b": 0.5}


def test_fusion_converts_positive_score_to_above_even_probability(tmp_path):
    fusion = KalshiSignalFusion(signal_names=["a"], state_path=str(tmp_path / "state.json"))
    fused = fusion.fuse([SignalValue("a", 1.0, 1.0, {})])

    assert fused.predicted_prob > 0.5
    assert fused.direction == "yes"


def test_weight_update_rewards_correct_signal(tmp_path):
    fusion = KalshiSignalFusion(signal_names=["a", "b"], state_path=str(tmp_path / "state.json"), min_samples=1)
    signals = [SignalValue("a", 1.0, 1.0, {}), SignalValue("b", -1.0, 1.0, {})]

    old_a = fusion.state.weights["a"]
    fusion.update_from_outcome(signals, yes_won=True)

    assert fusion.state.weights["a"] > old_a


def test_weight_update_uses_rolling_window(tmp_path):
    fusion = KalshiSignalFusion(signal_names=["a"], state_path=str(tmp_path / "state.json"), min_samples=1)
    signal = SignalValue("a", 1.0, 1.0, {})

    for _ in range(30):
        fusion.update_from_outcome([signal], yes_won=True)

    assert len(fusion.state.stats["a"].recent_correct) <= 50


def test_expected_value_subtracts_fee():
    ev_without_fee = expected_value(0.60, "yes", Decimal("0.50"), Decimal("0.00"))
    ev_with_fee = expected_value(0.60, "yes", Decimal("0.50"), Decimal("0.10"))

    assert ev_with_fee < ev_without_fee


def test_trade_intent_applies_ev_and_kelly_caps():
    config = KalshiRiskConfig(
        bankroll=Decimal("1000"),
        max_trade_fraction=Decimal("0.25"),
        kelly_multiplier=Decimal("0.50"),
        max_trade_dollars=Decimal("20"),
        ev_threshold=Decimal("0.01"),
        fee_per_contract=Decimal("0.01"),
    )

    intent = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=config,
    )

    assert intent.should_trade
    assert intent.side == "yes"
    assert intent.quantity > 0
    assert intent.kelly_fraction <= Decimal("0.25")


def test_trade_intent_rejects_boundary_no_liquidity_quotes():
    intent = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0"),
        no_ask=Decimal("1"),
        config=KalshiRiskConfig(),
    )

    assert not intent.should_trade
    assert intent.reason == "no_tradable_liquidity"


def test_trade_intent_fails_closed_without_exposure_data():
    intent = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=KalshiRiskConfig(fee_per_contract=Decimal("0.01")),
        current_exposure=None,
    )

    assert not intent.should_trade
    assert intent.reason == "unknown_exposure"


def test_trade_intent_fails_closed_without_daily_pnl():
    intent = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=KalshiRiskConfig(fee_per_contract=Decimal("0.01")),
        daily_pnl=None,
    )

    assert not intent.should_trade
    assert intent.reason == "unknown_daily_pnl"


def test_trade_intent_rejects_weak_edge_after_fees():
    intent = build_trade_intent(
        predicted_prob=0.54,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=KalshiRiskConfig(fee_per_contract=Decimal("0.01"), min_edge_after_fees=Decimal("0.05")),
    )

    assert not intent.should_trade
    assert intent.reason == "edge_below_threshold"


def test_trade_intent_rejects_bad_market_quality():
    wide = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=KalshiRiskConfig(fee_per_contract=Decimal("0.01"), max_spread_cents=4),
        spread_cents=Decimal("8"),
    )
    shallow = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=KalshiRiskConfig(fee_per_contract=Decimal("0.01"), min_top_depth=Decimal("10")),
        top_depth=Decimal("3"),
    )

    assert wide.reason == "spread_too_wide"
    assert shallow.reason == "insufficient_top_depth"


def test_trade_intent_reduces_size_after_drawdown():
    config = KalshiRiskConfig(
        bankroll=Decimal("1000"),
        max_trade_fraction=Decimal("0.25"),
        kelly_multiplier=Decimal("0.50"),
        max_trade_dollars=Decimal("1000"),
        max_daily_loss=Decimal("50"),
        max_total_exposure=Decimal("1000"),
        fee_per_contract=Decimal("0.01"),
        min_edge_after_fees=Decimal("0.01"),
    )
    normal = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=config,
        model_confidence=1.0,
        daily_pnl=Decimal("0"),
    )
    reduced = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=config,
        model_confidence=1.0,
        daily_pnl=Decimal("-30"),
    )

    assert reduced.quantity < normal.quantity
    assert reduced.sizing_multiplier == Decimal("0.5")


def test_trade_intent_dynamic_kelly_scaling():
    config = KalshiRiskConfig(
        bankroll=Decimal("1000"),
        max_trade_fraction=Decimal("0.25"),
        kelly_multiplier=Decimal("0.50"),
        max_trade_dollars=Decimal("1000"),
        max_daily_loss=Decimal("50"),
        max_total_exposure=Decimal("1000"),
        fee_per_contract=Decimal("0.01"),
        min_edge_after_fees=Decimal("0.01"),
    )
    high_conf = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=config,
        model_confidence=0.80,
        daily_pnl=Decimal("0"),
    )
    low_conf = build_trade_intent(
        predicted_prob=0.75,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=config,
        model_confidence=0.30,
        daily_pnl=Decimal("0"),
    )

    assert high_conf.quantity > low_conf.quantity


def test_trade_intent_early_window_noise_filter():
    config = KalshiRiskConfig(
        bankroll=Decimal("1000"),
        max_trade_fraction=Decimal("0.25"),
        kelly_multiplier=Decimal("0.50"),
        max_trade_dollars=Decimal("1000"),
        max_daily_loss=Decimal("50"),
        max_total_exposure=Decimal("1000"),
        fee_per_contract=Decimal("0.01"),
        min_edge_after_fees=Decimal("0.03"),
    )
    rejected = build_trade_intent(
        predicted_prob=0.55,
        yes_ask=Decimal("0.50"),
        no_ask=Decimal("0.52"),
        config=config,
        model_confidence=0.80,
        minutes_to_expiry=13.5,
        seconds_to_expiry=810,
        market_probability=0.51,
    )
    assert not rejected.should_trade
    assert rejected.reason == "early_window_low_edge"


def test_trade_intent_rejects_small_probability_gap():
    intent = build_trade_intent(
        predicted_prob=0.54,
        yes_ask=Decimal("0.45"),
        no_ask=Decimal("0.55"),
        market_probability=0.51,
        config=KalshiRiskConfig(
            fee_per_contract=Decimal("0.01"),
            min_edge_after_fees=Decimal("0.01"),
            min_probability_gap=Decimal("0.05"),
            min_model_confidence=Decimal("0.20"),
        ),
        model_confidence=0.80,
    )

    assert not intent.should_trade
    assert intent.reason == "probability_gap_too_small"


def test_trade_intent_rejects_tail_contract_probability():
    intent = build_trade_intent(
        predicted_prob=0.97,
        yes_ask=Decimal("0.90"),
        no_ask=Decimal("0.10"),
        market_probability=0.95,
        config=KalshiRiskConfig(
            fee_per_contract=Decimal("0.01"),
            min_edge_after_fees=Decimal("0.01"),
            min_probability_gap=Decimal("0.01"),
            min_model_confidence=Decimal("0.20"),
            min_market_side_prob=Decimal("0.08"),
            max_market_side_prob=Decimal("0.92"),
        ),
        model_confidence=0.90,
    )

    assert not intent.should_trade
    assert intent.reason == "tail_contract_probability"


def test_trade_intent_rejects_hard_late_entry_cutoff():
    intent = build_trade_intent(
        predicted_prob=0.80,
        yes_ask=Decimal("0.45"),
        no_ask=Decimal("0.55"),
        market_probability=0.40,
        config=KalshiRiskConfig(
            fee_per_contract=Decimal("0.01"),
            min_edge_after_fees=Decimal("0.01"),
            min_probability_gap=Decimal("0.01"),
            min_model_confidence=Decimal("0.20"),
            hard_late_entry_cutoff_seconds=180,
        ),
        model_confidence=0.90,
        seconds_to_expiry=120,
        minutes_to_expiry=2.0,
    )

    assert not intent.should_trade
    assert intent.reason == "late_entry_hard_cutoff"


def test_trade_intent_rejects_late_window_edge():
    intent = build_trade_intent(
        predicted_prob=0.70,
        yes_ask=Decimal("0.45"),
        no_ask=Decimal("0.55"),
        market_probability=0.45,
        config=KalshiRiskConfig(
            fee_per_contract=Decimal("0.01"),
            min_edge_after_fees=Decimal("0.01"),
            min_probability_gap=Decimal("0.01"),
            min_model_confidence=Decimal("0.20"),
            late_window_seconds=420,
            late_window_min_edge=Decimal("0.30"),
            hard_late_entry_cutoff_seconds=120,
        ),
        model_confidence=0.90,
        seconds_to_expiry=300,
        minutes_to_expiry=5.0,
    )

    assert not intent.should_trade
    assert intent.reason == "late_window_edge_too_small"


def test_trade_intent_rejects_late_window_confidence():
    intent = build_trade_intent(
        predicted_prob=0.80,
        yes_ask=Decimal("0.45"),
        no_ask=Decimal("0.55"),
        market_probability=0.40,
        config=KalshiRiskConfig(
            fee_per_contract=Decimal("0.01"),
            min_edge_after_fees=Decimal("0.01"),
            min_probability_gap=Decimal("0.01"),
            min_model_confidence=Decimal("0.20"),
            late_window_seconds=420,
            late_window_min_edge=Decimal("0.05"),
            late_window_min_confidence=Decimal("0.80"),
            hard_late_entry_cutoff_seconds=120,
        ),
        model_confidence=0.70,
        seconds_to_expiry=300,
        minutes_to_expiry=5.0,
    )

    assert not intent.should_trade
    assert intent.reason == "late_window_confidence_too_low"


def test_trade_intent_accepts_strong_trade_outside_late_cutoffs():
    intent = build_trade_intent(
        predicted_prob=0.80,
        yes_ask=Decimal("0.45"),
        no_ask=Decimal("0.55"),
        market_probability=0.40,
        config=KalshiRiskConfig(
            fee_per_contract=Decimal("0.01"),
            min_edge_after_fees=Decimal("0.05"),
            min_probability_gap=Decimal("0.05"),
            min_model_confidence=Decimal("0.50"),
            late_window_seconds=420,
            late_window_min_edge=Decimal("0.08"),
            late_window_min_confidence=Decimal("0.60"),
            hard_late_entry_cutoff_seconds=180,
            max_trade_dollars=Decimal("1000"),
            max_total_exposure=Decimal("1000"),
        ),
        model_confidence=0.90,
        seconds_to_expiry=600,
        minutes_to_expiry=10.0,
    )

    assert intent.should_trade
    assert intent.reason == "ok"

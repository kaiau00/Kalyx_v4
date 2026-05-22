"""
Expected-value, Kelly sizing, and risk gates for Kalshi binary contracts.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Literal, Optional


TradeSide = Literal["yes", "no"]


@dataclass
class KalshiRiskConfig:
    bankroll: Decimal = Decimal("1000.00")
    max_trade_fraction: Decimal = Decimal("0.25")
    kelly_multiplier: Decimal = Decimal("0.50")
    max_trade_dollars: Decimal = Decimal("25.00")
    max_daily_loss: Decimal = Decimal("50.00")
    max_total_exposure: Decimal = Decimal("100.00")
    ev_threshold: Decimal = Decimal("0.01")
    fee_per_contract: Decimal = Decimal("0.10")
    min_edge_after_fees: Decimal = Decimal("0.03")
    max_spread_cents: int = 6
    min_top_depth: Decimal = Decimal("5")
    min_model_confidence: Decimal = Decimal("0.20")
    late_trade_cutoff_seconds: int = 60
    drawdown_size_reduction_threshold: Decimal = Decimal("0.50")


@dataclass
class TradeIntent:
    should_trade: bool
    side: Optional[TradeSide]
    limit_price: Decimal
    limit_price_cents: int
    quantity: int
    expected_value: Decimal
    kelly_fraction: Decimal
    reason: str
    market_probability: Optional[Decimal] = None
    edge: Decimal = Decimal("0")
    edge_after_fees: Decimal = Decimal("0")
    sizing_multiplier: Decimal = Decimal("1")


def side_probability(predicted_yes_prob: float, side: TradeSide) -> Decimal:
    yes_prob = Decimal(str(predicted_yes_prob))
    return yes_prob if side == "yes" else Decimal("1") - yes_prob


def expected_value(predicted_yes_prob: float, side: TradeSide, price: Decimal, fee: Decimal) -> Decimal:
    win_prob = side_probability(predicted_yes_prob, side)
    payout_profit = Decimal("1") - price - fee
    loss = price + fee
    return (win_prob * payout_profit) - ((Decimal("1") - win_prob) * loss)


def capped_half_kelly(predicted_yes_prob: float, side: TradeSide, price: Decimal, config: KalshiRiskConfig) -> Decimal:
    win_prob = side_probability(predicted_yes_prob, side)
    net_profit = Decimal("1") - price - config.fee_per_contract
    loss = price + config.fee_per_contract
    if net_profit <= 0 or loss <= 0:
        return Decimal("0")
    odds = net_profit / loss
    full_kelly = (win_prob * odds - (Decimal("1") - win_prob)) / odds
    adjusted = full_kelly * config.kelly_multiplier
    return max(Decimal("0"), min(config.max_trade_fraction, adjusted))


def build_trade_intent(
    *,
    predicted_prob: float,
    yes_ask: Optional[Decimal],
    no_ask: Optional[Decimal],
    config: KalshiRiskConfig,
    current_exposure: Optional[Decimal] = Decimal("0"),
    daily_pnl: Optional[Decimal] = Decimal("0"),
    market_probability: Optional[float] = None,
    spread_cents: Optional[Decimal] = None,
    top_depth: Optional[Decimal] = None,
    model_confidence: Optional[float] = None,
    seconds_to_expiry: Optional[float] = None,
    minutes_to_expiry: Optional[float] = None,
) -> TradeIntent:
    if yes_ask is None or no_ask is None:
        return TradeIntent(False, None, Decimal("0"), 0, 0, Decimal("0"), Decimal("0"), "missing_quotes")
    if yes_ask <= 0 or yes_ask >= 1 or no_ask <= 0 or no_ask >= 1:
        return TradeIntent(False, None, Decimal("0"), 0, 0, Decimal("0"), Decimal("0"), "no_tradable_liquidity")
    if current_exposure is None:
        return TradeIntent(False, None, Decimal("0"), 0, 0, Decimal("0"), Decimal("0"), "unknown_exposure")
    if daily_pnl is None:
        return TradeIntent(False, None, Decimal("0"), 0, 0, Decimal("0"), Decimal("0"), "unknown_daily_pnl")

    yes_ev = expected_value(predicted_prob, "yes", yes_ask, config.fee_per_contract)
    no_ev = expected_value(predicted_prob, "no", no_ask, config.fee_per_contract)
    side: TradeSide = "yes" if yes_ev >= no_ev else "no"
    price = yes_ask if side == "yes" else no_ask
    ev = yes_ev if side == "yes" else no_ev
    side_prob = side_probability(predicted_prob, side)
    market_prob_dec = Decimal(str(market_probability)) if market_probability is not None else None
    if market_prob_dec is not None:
        market_side_prob = market_prob_dec if side == "yes" else Decimal("1") - market_prob_dec
        edge = side_prob - market_side_prob
    else:
        edge = side_prob - price
    edge_after_fees = ev

    mins_to_expiry = minutes_to_expiry if minutes_to_expiry is not None else (seconds_to_expiry / 60.0 if seconds_to_expiry is not None else None)

    if ev < config.ev_threshold:
        return TradeIntent(False, side, price, int(price * 100), 0, ev, Decimal("0"), "ev_below_threshold", market_prob_dec, edge, edge_after_fees)
    if edge_after_fees < config.min_edge_after_fees:
        return TradeIntent(False, side, price, int(price * 100), 0, ev, Decimal("0"), "edge_below_threshold", market_prob_dec, edge, edge_after_fees)
    if mins_to_expiry is not None and mins_to_expiry > 13.0 and edge_after_fees < config.min_edge_after_fees * Decimal("1.5"):
        return TradeIntent(False, side, price, int(price * 100), 0, ev, Decimal("0"), "early_window_low_edge", market_prob_dec, edge, edge_after_fees)
    if model_confidence is not None and Decimal(str(model_confidence)) < config.min_model_confidence:
        return TradeIntent(False, side, price, int(price * 100), 0, ev, Decimal("0"), "low_model_confidence", market_prob_dec, edge, edge_after_fees)
    if spread_cents is not None and spread_cents > Decimal(config.max_spread_cents):
        return TradeIntent(False, side, price, int(price * 100), 0, ev, Decimal("0"), "spread_too_wide", market_prob_dec, edge, edge_after_fees)
    if top_depth is not None and top_depth < config.min_top_depth:
        return TradeIntent(False, side, price, int(price * 100), 0, ev, Decimal("0"), "insufficient_top_depth", market_prob_dec, edge, edge_after_fees)
    if seconds_to_expiry is not None and seconds_to_expiry <= config.late_trade_cutoff_seconds and edge_after_fees < config.min_edge_after_fees * Decimal("2"):
        return TradeIntent(False, side, price, int(price * 100), 0, ev, Decimal("0"), "late_trade_edge_too_small", market_prob_dec, edge, edge_after_fees)
    if daily_pnl <= -config.max_daily_loss:
        return TradeIntent(False, side, price, int(price * 100), 0, ev, Decimal("0"), "daily_loss_cap", market_prob_dec, edge, edge_after_fees)
    if current_exposure >= config.max_total_exposure:
        return TradeIntent(False, side, price, int(price * 100), 0, ev, Decimal("0"), "exposure_cap", market_prob_dec, edge, edge_after_fees)

    kelly_fraction = capped_half_kelly(predicted_prob, side, price, config)

    if model_confidence is not None and model_confidence >= 0.70:
        kelly_scale = Decimal("0.65")
    elif model_confidence is not None and model_confidence >= 0.50:
        kelly_scale = Decimal("0.50")
    else:
        kelly_scale = Decimal("0.30")
    kelly_fraction = kelly_fraction * kelly_scale

    confidence_multiplier = Decimal(str(model_confidence)) if model_confidence is not None else Decimal("1")
    confidence_multiplier = max(Decimal("0"), min(Decimal("1"), confidence_multiplier))
    drawdown_multiplier = Decimal("1")
    drawdown_trigger = -(config.max_daily_loss * config.drawdown_size_reduction_threshold)
    if daily_pnl <= drawdown_trigger:
        drawdown_multiplier = Decimal("0.5")
    sizing_multiplier = confidence_multiplier * drawdown_multiplier
    trade_budget = min(
        config.bankroll * kelly_fraction * sizing_multiplier,
        config.max_trade_dollars,
        config.max_total_exposure - current_exposure,
    )
    contract_cost = price + config.fee_per_contract
    quantity = int((trade_budget / contract_cost).to_integral_value(rounding=ROUND_FLOOR)) if contract_cost > 0 else 0
    limit_price_cents = int((price * 100).to_integral_value(rounding=ROUND_FLOOR))

    if quantity <= 0:
        return TradeIntent(False, side, price, limit_price_cents, 0, ev, kelly_fraction, "quantity_zero", market_prob_dec, edge, edge_after_fees, sizing_multiplier)

    return TradeIntent(True, side, price, limit_price_cents, quantity, ev, kelly_fraction, "ok", market_prob_dec, edge, edge_after_fees, sizing_multiplier)

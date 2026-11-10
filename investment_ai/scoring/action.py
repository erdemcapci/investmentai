"""Risk-adjusted ranking keys, volatility-based trade plans and the action list.

The attractiveness scores answer "how good does this look?"; this layer
answers "how much of it, with what exit, and is the risk worth it?".  All
parameters are deterministic and configurable; none are fitted to outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from investment_ai.config import (
    ACTION_LIST_SIZE,
    ACTION_MAX_PER_SECTOR,
    EARNINGS_IMMINENT_DAYS,
    EARNINGS_IMMINENT_PENALTY,
    MAX_POSITION_PCT,
    PULLBACK_BONUS_CAP,
    PULLBACK_BONUS_MAX_DRAWDOWN_PCT,
    PULLBACK_BONUS_MIN_MOMENTUM,
    RISK_PENALTY_PER_POINT,
    TRADE_RISK_BUDGET_PCT,
)
from investment_ai.features.benchmark import REBOUND_RISK, RISK_OFF
from investment_ai.scoring.short_term import SHORT_TERM_WEIGHTS

SHORT_TERM_HOLDING_SESSIONS = 10
# The backtest replays only the momentum and earnings-drift pillars, whose
# weights renormalise to 87% and 13%; in the live score momentum is 65%.
# Scaling by their combined live weight keeps the bonus the same size relative
# to the momentum pillar as in the backtest (0.75 points per %, at most 11.25).
PULLBACK_LIVE_SCALE = SHORT_TERM_WEIGHTS["momentum"] + SHORT_TERM_WEIGHTS["earnings_drift"]
STOP_SIGMA_MULTIPLE = 2.0
REWARD_TO_RISK = 1.5
MIN_STOP_PCT, MAX_STOP_PCT = 3.0, 25.0


def pullback_bonus(row: dict, momentum_key: str = "short_rs_score") -> float:
    """Short-term points for a strong stock that has eased off its 20-day high.

    One point per % below the 20-day high in backtest score units, capped at
    ``PULLBACK_BONUS_CAP`` (the live score applies ``PULLBACK_LIVE_SCALE``),
    while the momentum pillar is strong, the price is above its 200-day
    average and the fall is under ``PULLBACK_BONUS_MAX_DRAWDOWN_PCT``.  In the
    5-year walk-forward backtest (2022-2026) strong stocks 5-20% below their
    20-day high in an intact uptrend recovered more than those at their high,
    with t-statistics of 2.2-2.8; beyond 20% the rebound was unreliable (t
    about 0.5) and another 15% fall within 10 sessions was 2-4 times as
    common, so such stocks earn nothing.  Shallow 3-8% dips alone did not help.
    """
    momentum = pd.to_numeric(row.get(momentum_key), errors="coerce")
    above_trend = pd.to_numeric(row.get("price_vs_ma200_pct"), errors="coerce")
    drawdown = pd.to_numeric(row.get("drawdown_from_20d_high_pct"), errors="coerce")
    if (pd.isna(momentum) or pd.isna(above_trend) or pd.isna(drawdown)
            or momentum < PULLBACK_BONUS_MIN_MOMENTUM or above_trend <= 0
            or -drawdown >= PULLBACK_BONUS_MAX_DRAWDOWN_PCT):
        return 0.0
    return float(min(max(-drawdown, 0.0), PULLBACK_BONUS_CAP))


def risk_adjusted_scores(row: dict) -> dict:
    """Deduct points for risk and imminent earnings; add the short-term pullback bonus."""
    risk = pd.to_numeric(row.get("risk_score"), errors="coerce")
    penalty = RISK_PENALTY_PER_POINT * max(risk - 50, 0) if pd.notna(risk) else 0.0
    days_to = pd.to_numeric(row.get("days_to_next_earnings"), errors="coerce")
    imminent = bool(pd.notna(days_to) and 0 <= days_to <= EARNINGS_IMMINENT_DAYS)
    bonus = pullback_bonus(row) * PULLBACK_LIVE_SCALE
    output = {"risk_penalty_points": penalty, "earnings_imminent_flag": imminent,
              "pullback_bonus_points": bonus}
    short_extra = (EARNINGS_IMMINENT_PENALTY if imminent else 0.0) - bonus
    for horizon, extra in (("long_term", 0.0), ("short_term", short_extra)):
        score = pd.to_numeric(row.get(f"{horizon}_score"), errors="coerce")
        output[f"{horizon}_risk_adjusted_score"] = (
            float(score - penalty - extra) if pd.notna(score) else np.nan
        )
    return output


def trade_plan(row: dict) -> dict:
    """Volatility-scaled stop, target and position size for a short-term idea.

    The stop sits two standard deviations of a ten-session move below the last
    completed close (about 12% for a typical 30%-volatility stock); the target
    is 1.5 times that distance above.  In the backtest a one-sigma stop cut the
    momentum list's return from +0.71% to +0.57% per trade, while this wider
    stop cost little (+0.66%) and still bounds the loss on a broken trade.  Position size
    risks ``TRADE_RISK_BUDGET_PCT`` of the portfolio at the stop, is capped at
    ``MAX_POSITION_PCT`` and is halved outside a risk-on market regime.
    """
    empty = {
        "stop_loss_pct": np.nan,
        "stop_loss_price": np.nan,
        "target_price": np.nan,
        "suggested_position_pct": np.nan,
        "holding_period_sessions": np.nan,
    }
    price = pd.to_numeric(row.get("current_price"), errors="coerce")
    volatility = pd.to_numeric(row.get("volatility_20d"), errors="coerce")
    if pd.isna(volatility):
        volatility = pd.to_numeric(row.get("volatility_60d"), errors="coerce")
    if pd.isna(price) or price <= 0 or pd.isna(volatility) or volatility <= 0:
        return empty
    daily = volatility / np.sqrt(252)
    stop = float(
        np.clip(
            STOP_SIGMA_MULTIPLE * daily * np.sqrt(SHORT_TERM_HOLDING_SESSIONS),
            MIN_STOP_PCT,
            MAX_STOP_PCT,
        )
    )
    size = min(TRADE_RISK_BUDGET_PCT / (stop / 100), MAX_POSITION_PCT)
    if row.get("market_regime") in {RISK_OFF, REBOUND_RISK}:
        size /= 2
    return {
        "stop_loss_pct": stop,
        "stop_loss_price": float(price * (1 - stop / 100)),
        "target_price": float(price * (1 + REWARD_TO_RISK * stop / 100)),
        "suggested_position_pct": float(size),
        "holding_period_sessions": SHORT_TERM_HOLDING_SESSIONS,
    }


def build_action_list(frame: pd.DataFrame) -> pd.DataFrame:
    """Short-term timing inside the long-term quality filter.

    Candidates must have a short-term rank and sit in the top half of the
    long-term ranking.  They are ordered by risk-adjusted short-term
    score with at most ``ACTION_MAX_PER_SECTOR`` names per sector.
    """
    lt_rank = pd.to_numeric(frame.get("long_term_rank"), errors="coerce")
    st_rank = pd.to_numeric(frame.get("short_term_rank"), errors="coerce")
    lt_count = int(lt_rank.notna().sum())
    if not lt_count or not st_rank.notna().any():
        return frame.iloc[0:0].assign(action_rank=pd.Series(dtype=float))
    # A stock whose price or stop cannot be trusted is never actionable.
    usable = frame.get("data_check_status", pd.Series("", index=frame.index)).ne("INCOMPLETE")
    eligible = frame[st_rank.notna() & lt_rank.le(lt_count / 2) & usable].copy()
    key = (
        "short_term_risk_adjusted_score"
        if "short_term_risk_adjusted_score" in eligible
        else "short_term_score"
    )
    eligible = eligible.sort_values(
        [key, "long_term_rank", "symbol"], ascending=[False, True, True], kind="stable"
    )
    sector = eligible.get("sector", pd.Series("", index=eligible.index)).fillna("")
    eligible = eligible[sector.groupby(sector).cumcount() < ACTION_MAX_PER_SECTOR]
    eligible = eligible.head(ACTION_LIST_SIZE).copy()
    eligible["action_rank"] = range(1, len(eligible) + 1)
    eligible["portfolio_weight_pct"] = portfolio_weights(eligible)
    return eligible


def portfolio_weights(frame: pd.DataFrame) -> pd.Series:
    """Equal-risk weights across the whole list, summing to at most 100%.

    Each name's weight is proportional to 1 / stop distance, so every position
    loses roughly the same amount at its stop.  Weights are normalised to 100%,
    capped at ``MAX_POSITION_PCT`` (the excess stays in cash) and halved for
    names outside a risk-on regime.
    """
    stop = pd.to_numeric(
        frame.get("stop_loss_pct", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    inverse = (1 / stop).where(stop.gt(0))
    if not inverse.notna().any():
        return pd.Series(np.nan, index=frame.index)
    weights = (inverse / inverse.sum() * 100).clip(upper=MAX_POSITION_PCT)
    regime = frame.get("market_regime", pd.Series("", index=frame.index))
    return weights.where(~regime.isin([RISK_OFF, REBOUND_RISK]), weights / 2)

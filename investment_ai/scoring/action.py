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
    RISK_PENALTY_PER_POINT,
    TRADE_RISK_BUDGET_PCT,
)
from investment_ai.features.benchmark import REBOUND_RISK, RISK_OFF

SHORT_TERM_HOLDING_SESSIONS = 10
STOP_SIGMA_MULTIPLE = 1.0
REWARD_TO_RISK = 1.5
MIN_STOP_PCT, MAX_STOP_PCT = 3.0, 15.0


def risk_adjusted_scores(row: dict) -> dict:
    """Deduct points for risk above neutral and for imminent earnings events."""
    risk = pd.to_numeric(row.get("risk_score"), errors="coerce")
    penalty = RISK_PENALTY_PER_POINT * max(risk - 50, 0) if pd.notna(risk) else 0.0
    days_to = pd.to_numeric(row.get("days_to_next_earnings"), errors="coerce")
    imminent = bool(pd.notna(days_to) and 0 <= days_to <= EARNINGS_IMMINENT_DAYS)
    output = {"risk_penalty_points": penalty, "earnings_imminent_flag": imminent}
    for horizon, extra in (("long_term", 0.0), ("short_term", EARNINGS_IMMINENT_PENALTY if imminent else 0.0)):
        score = pd.to_numeric(row.get(f"{horizon}_score"), errors="coerce")
        output[f"{horizon}_risk_adjusted_score"] = (
            float(score - penalty - extra) if pd.notna(score) else np.nan
        )
    return output


def trade_plan(row: dict) -> dict:
    """Volatility-scaled stop, target and position size for a short-term idea.

    The stop sits one standard deviation of a ten-session move below the last
    completed close (about 6% for a typical 30%-volatility stock); the target
    is 1.5 times that distance above.  Position size
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

    Candidates must have a credible short-term setup and sit in the top half
    of the long-term ranking.  They are ordered by risk-adjusted short-term
    score with at most ``ACTION_MAX_PER_SECTOR`` names per sector.
    """
    lt_rank = pd.to_numeric(frame.get("long_term_rank"), errors="coerce")
    st_rank = pd.to_numeric(frame.get("short_term_rank"), errors="coerce")
    lt_count = int(lt_rank.notna().sum())
    if not lt_count or not st_rank.notna().any():
        return frame.iloc[0:0].assign(action_rank=pd.Series(dtype=float))
    eligible = frame[st_rank.notna() & lt_rank.le(lt_count / 2)].copy()
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

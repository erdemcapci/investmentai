from __future__ import annotations
import numpy as np
import pandas as pd
from investment_ai.features.benchmark import REBOUND_RISK
from investment_ai.features.technical import (
    event_timing_score,
    model_rs,
    setup_scores,
)
from investment_ai.scoring.common import (
    INSUFFICIENT_DATA,
    NEUTRAL,
    curve,
    safe_nanmean,
    weighted,
)

CREDIBLE_SETUPS = {"PULLBACK", "BREAKOUT", "MOMENTUM_CONTINUATION", "MIXED"}
# Model 3.3.0 weights follow the 5-year walk-forward backtest (2021-2026):
# a top-20 momentum list earned +0.71% excess per 10 sessions net of costs
# (t = 2.6), while blending in the setup, volume and technical pillars cut that
# to +0.06%.  Those pillars are still computed and exported as diagnostics but
# carry no weight.  Expectations and short interest cannot be replayed and keep
# literature-based weights; earnings drift (IC t = 2.0) is kept small because a
# larger share diluted momentum.
SHORT_TERM_WEIGHTS = {
    "momentum": 0.65,
    "expectations": 0.20,
    "earnings_drift": 0.10,
    "positioning": 0.05,
}


def momentum_pillar(row: dict) -> tuple[float, float]:
    """Skip-month momentum led by the 12-1 month return.

    In the 5-year walk-forward backtest (2021-2026, weekly) 12-1 and 6-1 month
    momentum were the only price signals with a consistently positive IC; the
    20-day reversal and 20-day strength terms were indistinguishable from zero.
    The 60-day term keeps coverage for listings with under a year of history.
    """
    score, coverage, _ = weighted(
        {
            "mom_12_1": model_rs(row, "mom_12_1"),
            "mom_6_1": model_rs(row, "mom_6_1"),
            "rs60": model_rs(row, "60d"),
        },
        {"mom_12_1": 0.50, "mom_6_1": 0.30, "rs60": 0.20},
        0.20,
    )
    return score, coverage


def volume_pillar(row: dict) -> float:
    """Volume surprise signed by price direction (accumulation vs distribution)."""
    relative, move = row.get("relative_volume_5d"), row.get("return_5d_pct")
    if pd.isna(relative) or pd.isna(move):
        relative, move = row.get("relative_volume_1d"), row.get("return_1d_pct")
    if relative is None or move is None or pd.isna(relative) or pd.isna(move):
        return np.nan
    signed = (relative - 1) * np.sign(move)
    return curve(signed, [(-1, 10), (-0.3, 35), (0, 50), (0.3, 65), (1, 85), (2.5, 70)])


def technical_pillar(row: dict) -> float:
    return safe_nanmean(
        [
            curve(
                row.get("price_vs_ma20_pct"), [(-15, 0), (0, 60), (8, 100), (25, 40)]
            ),
            curve(
                row.get("price_vs_ma50_pct"), [(-20, 0), (0, 60), (15, 100), (40, 40)]
            ),
        ]
    )


def positioning_pillar(row: dict) -> float:
    """Short-interest level and change; crowded or rising shorts are negative."""
    level = curve(
        row.get("short_interest_pct_float"),
        [(0, 60), (2, 55), (5, 45), (10, 30), (20, 15)],
    )
    change = curve(
        row.get("short_interest_change_pct"), [(-30, 80), (0, 50), (30, 20)]
    )
    return weighted({"level": level, "change": change}, {"level": 0.5, "change": 0.5})[0]


def score_short_term(row: dict) -> dict:
    setup = setup_scores(row)
    relative_strength, rs_coverage = momentum_pillar(row)
    volume = volume_pillar(row)
    technical = technical_pillar(row)
    event = event_timing_score(
        row.get("days_to_next_earnings"), row.get("days_since_last_earnings")
    )
    drift = row.get("earnings_drift_score", np.nan)
    positioning = positioning_pillar(row)
    pillars = {
        "momentum": relative_strength,
        "expectations": row.get("expectations_short_score"),
        "earnings_drift": drift,
        "positioning": positioning,
    }
    weights = dict(SHORT_TERM_WEIGHTS)
    if row.get("market_regime") == REBOUND_RISK:
        # Momentum crashes cluster in sharp rebounds after market declines.
        weights["momentum"] /= 2
    score, coverage, status = weighted(pillars, weights, 0.60, neutral_fill=NEUTRAL)
    core_ok = (
        pd.notna(relative_strength)
        and pd.notna(row.get("expectations_short_score"))
        and row.get("expectations_short_coverage", 0) >= 0.60
    )
    # The setup label no longer gates ranking: in the backtest, restricting
    # the list to credible setups halved the momentum list's excess return.
    if not core_ok:
        score, status = np.nan, INSUFFICIENT_DATA
    return {
        **setup,
        "short_rs_score": relative_strength,
        "short_rs_coverage": rs_coverage,
        "volume_confirmation_score": volume,
        "technical_trend_score": technical,
        "positioning_score": positioning,
        # Event timing is a risk input only; it no longer contributes to alpha.
        "event_timing_score": event,
        "short_term_score": score,
        "short_term_coverage": coverage,
        "short_term_status": status,
    }

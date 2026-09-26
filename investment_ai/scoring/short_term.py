from __future__ import annotations
import numpy as np
import pandas as pd
from investment_ai.features.technical import event_timing_score, setup_scores
from investment_ai.scoring.common import INSUFFICIENT_DATA, curve, weighted, safe_nanmean

NO_CREDIBLE_SETUP = "NO_CREDIBLE_SETUP"
CREDIBLE_SETUPS = {"PULLBACK", "BREAKOUT", "MOMENTUM_CONTINUATION", "MIXED"}


def score_short_term(row: dict) -> dict:
    setup = setup_scores(row)
    relative_strength, rs_coverage, _ = weighted(
        {
            "rs20": row.get("rs_20d_percentile"),
            "rs60": row.get("rs_60d_percentile"),
            "rs126": row.get("rs_126d_percentile"),
        },
        {"rs20": 0.50, "rs60": 0.35, "rs126": 0.15},
        0.60,
    )
    volume = curve(
        row.get("relative_volume_1d"), [(0.3, 20), (1, 55), (1.8, 100), (4, 70)]
    )
    technical = safe_nanmean(
        [
            curve(
                row.get("price_vs_ma20_pct"), [(-15, 0), (0, 60), (8, 100), (25, 40)]
            ),
            curve(
                row.get("price_vs_ma50_pct"), [(-20, 0), (0, 60), (15, 100), (40, 40)]
            ),
        ]
    )
    event = event_timing_score(
        row.get("days_to_next_earnings"), row.get("days_since_last_earnings")
    )
    pillars = {
        "relative_strength": relative_strength,
        "setup": setup["setup_quality_score"],
        "expectations": row.get("expectations_short_score"),
        "volume": volume,
        "technical": technical,
        "event": event,
    }
    score, coverage, status = weighted(
        pillars,
        {
            "relative_strength": 0.20,
            "setup": 0.25,
            "expectations": 0.20,
            "volume": 0.10,
            "technical": 0.15,
            "event": 0.10,
        },
        0.70,
    )
    core_ok = (
        pd.notna(relative_strength)
        and rs_coverage >= 0.60
        and pd.notna(setup["setup_quality_score"])
        and setup["setup_coverage"] >= 0.70
        and pd.notna(row.get("expectations_short_score"))
        and row.get("expectations_short_coverage", 0) >= 0.60
    )
    if setup["short_term_setup"] not in CREDIBLE_SETUPS:
        score, status = np.nan, NO_CREDIBLE_SETUP
    elif not core_ok:
        score, status = np.nan, INSUFFICIENT_DATA
    return {
        **setup,
        "short_rs_score": relative_strength,
        "short_rs_coverage": rs_coverage,
        "volume_confirmation_score": volume,
        "technical_trend_score": technical,
        "event_timing_score": event,
        "short_term_score": score,
        "short_term_coverage": coverage,
        "short_term_status": status,
    }

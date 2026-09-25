from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from investment_ai.scoring.common import INSUFFICIENT_DATA, RANKED, curve, weighted, safe_nanmean


def market_price_risk_score(row: dict) -> float:
    return safe_nanmean([
        curve(row.get("volatility_60d"), [(10, 10), (30, 45), (60, 85), (100, 100)]),
        curve(row.get("downside_volatility"), [(8, 10), (25, 45), (55, 90), (90, 100)]),
        curve(abs(row.get("max_drawdown_1y", np.nan)), [(5, 10), (20, 45), (50, 100)]),
    ])


def _age_hours(value):
    try:
        timestamp = datetime.fromisoformat(str(value))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - timestamp).total_seconds() / 3600, 0)
    except (ValueError, TypeError):
        return np.nan


def risk_and_confidence(row: dict) -> dict:
    market = row.get("market_price_risk_score")
    if pd.isna(market):
        market = market_price_risk_score(row)
    if row.get("is_financial", False):
        balance = np.nan
    else:
        balance = safe_nanmean(
            [
                curve(row.get("debt_to_equity"), [(0, 5), (1, 30), (3, 75), (6, 100)]),
                curve(row.get("net_debt_to_ebitda"), [(-1, 0), (2, 35), (5, 90)]),
                curve(row.get("interest_coverage"), [(0, 100), (3, 55), (10, 10)]),
            ]
        )
        if row.get("negative_equity_flag") or row.get("negative_ebitda_flag") or row.get("negative_operating_profit_flag"):
            balance = 100.0
    event = 100 - row.get("event_timing_score", np.nan)
    target_dispersion = row.get("target_dispersion_pct")
    disagreement, _, _ = weighted(
        {
            "target": curve(target_dispersion, [(0, 0), (30, 55), (80, 100)]),
            "eps": curve(
                row.get("eps_plus_1y_dispersion_pct"), [(5, 5), (30, 55), (80, 100)]
            ),
            "ratings": curve(
                row.get("negative_rating_pct_current"), [(0, 0), (20, 50), (50, 100)]
            ),
        },
        {"target": 0.50, "eps": 0.25, "ratings": 0.25},
        0,
    )
    liquidity = curve(
        row.get("average_dollar_volume_20d"), [(1e6, 100), (1e7, 55), (1e8, 10)]
    )
    risk, risk_coverage, _ = weighted(
        {
            "market": market,
            "balance": balance,
            "event": event,
            "disagreement": disagreement,
            "liquidity": liquidity,
        },
        {
            "market": 0.30,
            "balance": 0.25,
            "event": 0.15,
            "disagreement": 0.20,
            "liquidity": 0.10,
        },
        0.60,
    )
    ages = {
        tier: _age_hours(row.get(f"{tier}_fetched_at_utc"))
        for tier in ("analyst", "valuation", "fundamentals")
    }
    freshness = safe_nanmean(
        [curve(age, [(0, 100), (24, 90), (168, 50), (720, 0)]) for age in ages.values()]
    )
    coverage = (
        safe_nanmean(
            [
                row.get("long_term_coverage", np.nan),
                row.get("short_term_coverage", np.nan),
                row.get("expectations_coverage", np.nan),
            ]
        )
        * 100
    )
    analyst = curve(row.get("rating_count"), [(0, 20), (5, 50), (15, 85), (30, 100)])
    confidence, _, _ = weighted(
        {
            "coverage": coverage,
            "analyst": analyst,
            "freshness": freshness,
            "provider": row.get("provider_success", 0) * 100,
        },
        {"coverage": 0.45, "analyst": 0.20, "freshness": 0.20, "provider": 0.15},
        0,
    )
    return {
        "risk_score": risk,
        "risk_coverage": risk_coverage,
        "risk_status": RANKED if pd.notna(risk) else INSUFFICIENT_DATA,
        "confidence_score": confidence,
        "market_risk_score": market,
        "market_price_risk_score": market,
        "balance_sheet_risk_score": balance,
        "analyst_disagreement_score": disagreement,
        "analyst_freshness_age_hours": ages["analyst"],
        "valuation_freshness_age_hours": ages["valuation"],
        "fundamental_freshness_age_hours": ages["fundamentals"],
    }

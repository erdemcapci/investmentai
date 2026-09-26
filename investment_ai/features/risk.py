from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from investment_ai.scoring.common import (
    INSUFFICIENT_DATA,
    RANKED,
    curve,
    weighted,
    safe_nanmean,
)
from investment_ai.status import (
    ERROR,
    FRESH,
    FRESH_CACHE,
    FRESH_PROVIDER,
    INSUFFICIENT,
    PARTIAL,
    STALE_FALLBACK,
)

COMPONENT_STATUS_MULTIPLIERS = {
    FRESH_PROVIDER: 1.00,
    FRESH_CACHE: 0.95,
    FRESH: 1.00,
    STALE_FALLBACK: 0.50,
    PARTIAL: 0.40,
    ERROR: 0.00,
    INSUFFICIENT: 0.00,
}
ANALYST_COMPONENTS = (
    "targets",
    "recommendations",
    "eps_trend",
    "eps_revisions",
    "revenue_estimate",
    "earnings_estimate",
    "rating_actions",
    "earnings_history",
    "earnings_dates",
)


def market_price_risk_score(row: dict) -> float:
    return safe_nanmean(
        [
            curve(
                row.get("volatility_60d"), [(10, 10), (30, 45), (60, 85), (100, 100)]
            ),
            curve(
                row.get("downside_volatility"), [(8, 10), (25, 45), (55, 90), (90, 100)]
            ),
            curve(
                abs(row.get("max_drawdown_1y", np.nan)), [(5, 10), (20, 45), (50, 100)]
            ),
        ]
    )


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
        balance_reason = "FINANCIAL_SECTOR_GENERIC_METRICS_NOT_APPLICABLE"
    else:
        balance = safe_nanmean(
            [
                curve(row.get("debt_to_equity"), [(0, 5), (1, 30), (3, 75), (6, 100)]),
                curve(row.get("net_debt_to_ebitda"), [(-1, 0), (2, 35), (5, 90)]),
                curve(row.get("interest_coverage"), [(0, 100), (3, 55), (10, 10)]),
            ]
        )
        net_debt = pd.to_numeric(row.get("net_debt"), errors="coerce")
        net_cash = bool(row.get("net_cash_flag")) or (
            pd.notna(net_debt) and net_debt <= 0
        )
        if row.get("negative_equity_flag"):
            balance = 100.0
            balance_reason = "NEGATIVE_EQUITY"
        elif row.get("negative_ebitda_flag") and pd.notna(net_debt) and net_debt > 0:
            balance = max(balance, 95.0) if pd.notna(balance) else 95.0
            balance_reason = "NEGATIVE_EBITDA_WITH_POSITIVE_NET_DEBT"
        elif row.get("negative_ebitda_flag") and net_cash:
            balance_reason = "NEGATIVE_EBITDA_WITH_NET_CASH_METRIC_BASED"
        elif (
            row.get("negative_operating_profit_flag")
            and pd.notna(net_debt)
            and net_debt > 0
        ):
            balance = max(balance, 90.0) if pd.notna(balance) else 90.0
            balance_reason = "NEGATIVE_OPERATING_PROFIT_WITH_POSITIVE_NET_DEBT"
        elif row.get("negative_operating_profit_flag"):
            balance_reason = "NEGATIVE_OPERATING_PROFIT_METRIC_BASED"
        else:
            balance_reason = "ORDINARY_METRIC_BASED"
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
    status_values = [
        row.get(f"{name}_cache_status", INSUFFICIENT) for name in ANALYST_COMPONENTS
    ]
    system_statuses = status_values + [
        row.get("valuation_cache_status", INSUFFICIENT),
        row.get("fundamental_cache_status", INSUFFICIENT),
        row.get("price_data_status", INSUFFICIENT),
    ]
    status_scores = [
        COMPONENT_STATUS_MULTIPLIERS.get(value, 0.0) for value in system_statuses
    ]
    freshness = safe_nanmean(status_scores) * 100
    # Completeness is presence/usable-state only; it deliberately ignores age.
    provider_completeness = safe_nanmean(
        [
            100.0 if value not in {ERROR, INSUFFICIENT, None} else 0.0
            for value in system_statuses
        ]
    )
    analyst_status_scores = [
        COMPONENT_STATUS_MULTIPLIERS.get(value, 0.0) for value in status_values
    ]
    analyst_fresh_count = sum(score >= 0.95 for score in analyst_status_scores)
    analyst_stale_count = sum(0 < score < 0.95 for score in analyst_status_scores)
    analyst_error_count = sum(status == ERROR for status in status_values)
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
            "provider": provider_completeness,
        },
        {"coverage": 0.40, "analyst": 0.20, "freshness": 0.20, "provider": 0.20},
        0,
    )
    # Reliable CET1/NPL/NIM/regulatory-capital inputs are unavailable; do not
    # invent proxies, and make that sector-specific limitation visible here.
    if row.get("is_financial", False) and pd.notna(confidence):
        confidence *= 0.90
    return {
        "risk_score": risk,
        "risk_coverage": risk_coverage,
        "risk_status": RANKED if pd.notna(risk) else INSUFFICIENT_DATA,
        "confidence_score": confidence,
        "market_risk_score": market,
        "market_price_risk_score": market,
        "balance_sheet_risk_score": balance,
        "balance_sheet_risk_reason": balance_reason,
        "analyst_disagreement_score": disagreement,
        "analyst_freshness_age_hours": ages["analyst"],
        "valuation_freshness_age_hours": ages["valuation"],
        "fundamental_freshness_age_hours": ages["fundamentals"],
        "analyst_component_freshness_score": safe_nanmean(analyst_status_scores) * 100,
        "freshness_score": freshness,
        "provider_completeness_score": provider_completeness,
        "analyst_component_fresh_count": analyst_fresh_count,
        "analyst_component_stale_count": analyst_stale_count,
        "analyst_component_error_count": analyst_error_count,
    }

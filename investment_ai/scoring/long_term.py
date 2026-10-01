from __future__ import annotations
import numpy as np
import pandas as pd
from investment_ai.features.technical import model_rs
from investment_ai.scoring.common import (
    INSUFFICIENT_DATA,
    NEUTRAL,
    curve,
    weighted,
    safe_nanmean,
)

# Metrics also ranked within sector and region; the peer percentile is blended
# 50/50 with the absolute curve so that sector norms (software margins versus
# retail margins, US versus European growth) do not dominate the ranking.
RELATIVE_METRICS = (
    "fcf_margin_pct",
    "operating_margin_pct",
    "capital_returns_pct",
    "return_on_equity",
    "return_on_assets",
    "net_margin_pct",
    "revenue_growth_yoy",
    "forward_revenue_growth",
    "forward_eps_growth",
)


def _relative(row: dict, absolute: float, metric: str) -> float:
    """Blend an absolute curve score with the sector-region peer percentile."""
    return safe_nanmean([absolute, row.get(f"{metric}_sector_pct", np.nan)])


def _trend_rs(row: dict, primary: str, fallback: str) -> float:
    value = model_rs(row, primary)
    return model_rs(row, fallback) if pd.isna(value) else value


def long_trend_pillar(row: dict) -> tuple[float, float]:
    """Skip-month momentum (6-1 and 12-1) plus distance from the 200-day average."""
    trend = {
        "rs126": _trend_rs(row, "mom_6_1", "126d"),
        "rs252": _trend_rs(row, "mom_12_1", "252d"),
        "ma200": curve(
            row.get("price_vs_ma200_pct"), [(-30, 0), (0, 55), (20, 100), (50, 60)]
        ),
    }
    score, coverage, _ = weighted(
        trend, {"rs126": 0.35, "rs252": 0.35, "ma200": 0.30}, 0.60
    )
    return score, coverage


def score_long_term(row: dict) -> dict:
    financial = bool(row.get("is_financial", False))
    if financial:
        quality = {
            "roe": _relative(row, curve(
                row.get("return_on_equity"), [(-10, 0), (0, 30), (15, 75), (30, 100)]
            ), "return_on_equity"),
            "roa": _relative(row, curve(
                row.get("return_on_assets"), [(-2, 0), (0, 35), (2, 70), (5, 100)]
            ), "return_on_assets"),
            "net_margin": _relative(row, curve(
                row.get("net_margin_pct"), [(-10, 0), (0, 35), (20, 100)]
            ), "net_margin_pct"),
            "earnings_growth": curve(
                row.get("earnings_growth_yoy"), [(-30, 0), (0, 45), (30, 100)]
            ),
            "revenue_growth": curve(
                row.get("revenue_growth_yoy"), [(-20, 0), (0, 45), (20, 100)]
            ),
        }
        quality_weights = {
            "roe": 0.25,
            "roa": 0.20,
            "net_margin": 0.20,
            "earnings_growth": 0.20,
            "revenue_growth": 0.15,
        }
    else:
        quality = {
            "fcf_margin": _relative(row, curve(
                row.get("fcf_margin_pct"), [(-10, 0), (0, 35), (10, 70), (25, 100)]
            ), "fcf_margin_pct"),
            "operating_margin": _relative(row, curve(
                row.get("operating_margin_pct"),
                [(-10, 0), (0, 30), (15, 75), (35, 100)],
            ), "operating_margin_pct"),
            "margin_trend": curve(
                row.get("operating_margin_change_1y"), [(-10, 0), (0, 50), (10, 100)]
            ),
            "fcf_growth": curve(
                row.get("fcf_growth_yoy"), [(-50, 0), (0, 50), (50, 100)]
            ),
            "cash_conversion": curve(
                row.get("cash_conversion"), [(0, 0), (0.8, 70), (1.5, 100), (3, 65)]
            ),
            "returns": _relative(row, curve(
                safe_nanmean(
                    [row.get("return_on_equity", np.nan), row.get("roic", np.nan)]
                ),
                [(-10, 0), (0, 30), (15, 75), (30, 100)],
            ), "capital_returns_pct"),
        }
        quality_weights = {
            "fcf_margin": 0.20,
            "operating_margin": 0.15,
            "margin_trend": 0.15,
            "fcf_growth": 0.15,
            "cash_conversion": 0.15,
            "returns": 0.20,
        }
    quality_score, quality_coverage, quality_status = weighted(
        quality, quality_weights, 0.60
    )
    historical_growth = weighted(
        {
            "yoy": _relative(
                row,
                curve(row.get("revenue_growth_yoy"), [(-20, 0), (0, 40), (30, 100)]),
                "revenue_growth_yoy",
            ),
            "cagr": curve(row.get("revenue_cagr_3y"), [(-10, 0), (0, 40), (20, 100)]),
        },
        {"yoy": 0.60, "cagr": 0.40},
        0,
    )[0]
    growth = {
        "forward_revenue": _relative(row, curve(
            row.get("forward_revenue_growth"), [(-0.2, 0), (0, 40), (0.3, 100)]
        ), "forward_revenue_growth"),
        "forward_eps": _relative(row, curve(
            row.get("forward_eps_growth"), [(-0.3, 0), (0, 40), (0.4, 100)]
        ), "forward_eps_growth"),
        "historical_revenue": historical_growth,
        "latest_quarter": curve(
            row.get("revenue_growth_latest_quarter_yoy"), [(-20, 0), (0, 40), (30, 100)]
        ),
        "acceleration": curve(
            row.get("growth_acceleration"), [(-20, 0), (0, 50), (20, 100)]
        ),
    }
    growth_score, growth_coverage, growth_status = weighted(
        growth,
        {
            "forward_revenue": 0.30,
            "forward_eps": 0.30,
            "historical_revenue": 0.20,
            "latest_quarter": 0.10,
            "acceleration": 0.10,
        },
        0.60,
    )
    if financial:
        valuation = {
            "forward_pe": row.get("forward_pe_peer_percentile"),
            "price_book": row.get("price_book_peer_percentile"),
            "peg": row.get("peg_peer_percentile"),
            "own_pe": row.get("forward_pe_own_history_percentile"),
            "own_pb": row.get("price_book_own_history_percentile"),
        }
        valuation_weights = {
            "forward_pe": 0.25,
            "price_book": 0.25,
            "peg": 0.15,
            "own_pe": 0.20,
            "own_pb": 0.15,
        }
    else:
        valuation = {
            "forward_pe": row.get("forward_pe_peer_percentile"),
            "ev_ebitda": row.get("ev_ebitda_peer_percentile"),
            "fcf_yield": row.get("fcf_yield_peer_percentile"),
            "peg": row.get("peg_peer_percentile"),
            "own_history": row.get("forward_pe_own_history_percentile"),
        }
        valuation_weights = {
            "forward_pe": 0.25,
            "ev_ebitda": 0.25,
            "fcf_yield": 0.20,
            "peg": 0.15,
            "own_history": 0.15,
        }
    valuation_score, valuation_coverage, valuation_status = weighted(
        valuation, valuation_weights, 0.50
    )
    long_trend, trend_coverage = long_trend_pillar(row)
    if financial:
        safety = {
            "profitability": curve(
                row.get("net_margin_pct"), [(-10, 0), (0, 40), (20, 100)]
            ),
            "earnings": curve(
                row.get("earnings_growth_yoy"), [(-50, 0), (0, 55), (30, 100)]
            ),
            "price_risk": 100 - row.get("market_price_risk_score", np.nan),
        }
        safety_weights = {"profitability": 0.40, "earnings": 0.30, "price_risk": 0.30}
    else:
        safety = {
            "leverage": curve(
                row.get("debt_to_equity"), [(0, 100), (1, 75), (3, 25), (6, 0)]
            ),
            "net_debt": curve(
                row.get("net_debt_to_ebitda"), [(-1, 100), (1, 80), (3, 40), (6, 0)]
            ),
            "coverage": curve(
                row.get("interest_coverage"), [(0, 0), (3, 50), (10, 100)]
            ),
        }
        safety_weights = {"leverage": 0.35, "net_debt": 0.35, "coverage": 0.30}
    safety_score, _, _ = weighted(safety, safety_weights, 0.40)
    pillars = {
        "quality": quality_score,
        "growth": growth_score,
        "valuation": valuation_score,
        "expectations": row.get("expectations_long_score"),
        "long_trend": long_trend,
        "financial_safety": safety_score,
    }
    score, coverage, status = weighted(
        pillars,
        {
            "quality": 0.25,
            "growth": 0.20,
            "valuation": 0.20,
            "expectations": 0.20,
            "long_trend": 0.10,
            "financial_safety": 0.05,
        },
        0.70,
        neutral_fill=NEUTRAL,
    )
    if any(
        pd.isna(pillars[key])
        for key in ("quality", "growth", "valuation", "expectations")
    ):
        score, status = np.nan, INSUFFICIENT_DATA
    return {
        "quality_score": quality_score,
        "quality_coverage": quality_coverage,
        "quality_status": quality_status,
        "growth_score": growth_score,
        "growth_coverage": growth_coverage,
        "growth_status": growth_status,
        "valuation_score": valuation_score,
        "valuation_coverage": valuation_coverage,
        "valuation_status": valuation_status,
        "long_trend_score": long_trend,
        "long_trend_coverage": trend_coverage,
        "financial_safety_score": safety_score,
        "long_term_score": score,
        "long_term_coverage": coverage,
        "long_term_status": status,
    }

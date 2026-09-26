"""Merge provider inputs and calculate deterministic scores."""

from __future__ import annotations
import numpy as np
import pandas as pd
from investment_ai.features.analyst import expectations_score
from investment_ai.features.risk import market_price_risk_score, risk_and_confidence
from investment_ai.status import (
    ERROR,
    FRESH,
    FRESH_CACHE,
    FRESH_PROVIDER,
    PARTIAL,
    STALE,
    STALE_FALLBACK,
)
from investment_ai.features.technical import add_relative_strength
from investment_ai.features.benchmark import add_benchmark_relative_strength
from investment_ai.features.valuation import add_peer_percentiles
from investment_ai.scoring.common import curve, weighted
from investment_ai.scoring.long_term import score_long_term
from investment_ai.scoring.ranking import rank_results
from investment_ai.scoring.short_term import score_short_term

SECTOR_MAP = {
    "technology": "Technology",
    "information technology": "Technology",
    "tech": "Technology",
    "financials": "Financials",
    "financial services": "Financials",
    "banks": "Financials",
    "insurance": "Financials",
    "consumer cyclical": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "basic materials": "Materials",
    "industrials": "Industrials",
    "energy": "Energy",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "communication services": "Communication Services",
    "consumer defensive": "Consumer Staples",
    "consumer staples": "Consumer Staples",
}


def normalize_sector(value):
    if pd.isna(value):
        return np.nan
    return SECTOR_MAP.get(str(value).strip().lower(), str(value).strip())


def target_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate upside independently; only dispersion requires a valid range."""
    selected = frame.get("target_median", pd.Series(np.nan, index=frame.index)).where(
        frame.get("target_median", pd.Series(np.nan, index=frame.index)) > 0
    ).combine_first(frame.get("target_mean", pd.Series(np.nan, index=frame.index)).where(
        frame.get("target_mean", pd.Series(np.nan, index=frame.index)) > 0
    ))
    low = pd.to_numeric(frame.get("target_low", pd.Series(np.nan, index=frame.index)), errors="coerce")
    high = pd.to_numeric(frame.get("target_high", pd.Series(np.nan, index=frame.index)), errors="coerce")
    price = pd.to_numeric(frame.get("current_price", pd.Series(np.nan, index=frame.index)), errors="coerce")
    valid_range = low.notna() & high.notna() & low.le(high)
    return pd.DataFrame({
        "selected_target": selected,
        "target_range_valid": valid_range,
        "target_upside_pct": np.where(selected.gt(0) & price.gt(0), (selected / price - 1) * 100, np.nan),
        "target_dispersion_pct": np.where(valid_range & selected.gt(0), (high - low) / selected.abs() * 100, np.nan),
    }, index=frame.index)


def _history_scores_v31(row):
    target_changes = {}
    for days, weight in ((7, 0.45), (30, 0.35), (90, 0.20)):
        value = row.get(f"target_median_change_{days}d_pct")
        if pd.isna(value):
            value = row.get(f"target_mean_change_{days}d_pct")
        value = np.nan if value is None else value
        target_changes[str(days)] = curve(
            np.clip(value, -100, 100), [(-30, 0), (0, 50), (15, 80), (40, 100)]
        )
    target_score = weighted(target_changes, {"7": 0.45, "30": 0.35, "90": 0.20}, 0)[0]
    def revenue_curve(period, days):
        return curve(row.get(f"revenue_{period}_change_{days}d_pct"), [(-20,0),(0,50),(20,100)])
    q0 = weighted({"7": revenue_curve("0q",7), "30": revenue_curve("0q",30)}, {"7":.60,"30":.40}, 0)[0]
    q1 = weighted({"7": revenue_curve("plus_1q",7), "30": revenue_curve("plus_1q",30)}, {"7":.50,"30":.50}, 0)[0]
    revenue_short = weighted({"0q":q0,"plus_1q":q1},{"0q":.55,"plus_1q":.45},0)[0]
    y0 = weighted({"30":revenue_curve("0y",30),"90":revenue_curve("0y",90)}, {"30":.7,"90":.3},0)[0]
    y1 = weighted({"30":revenue_curve("plus_1y",30),"90":revenue_curve("plus_1y",90)}, {"30":.7,"90":.3},0)[0]
    revenue_long = weighted({"0y":y0,"plus_1y":y1},{"0y":.40,"plus_1y":.60},0)[0]
    return target_score, revenue_short, revenue_long


def _history_scores(row):
    """Backward-compatible target/long-revenue pair."""
    target, _, long_revenue = _history_scores_v31(row)
    return target, long_revenue


LT_DRIVER_SPECS = (
    ("Business quality", "Quality", "quality_score", .25),
    ("Growth", "Growth", "growth_score", .20),
    ("Valuation vs peers", "Valuation", "valuation_score", .20),
    ("Long expectations", "Long Expectations", "expectations_long_score", .20),
    ("Long trend", "Long Trend", "long_trend_score", .10),
    ("Financial safety", "Financial Safety", "financial_safety_score", .05),
)
ST_DRIVER_SPECS = (
    ("Relative strength", "Relative Strength", "short_rs_score", .20),
    ("Setup quality", "Setup Quality", "setup_quality_score", .25),
    ("Short expectations", "Short Expectations", "expectations_short_score", .20),
    ("Volume", "Volume", "volume_confirmation_score", .10),
    ("Technical trend", "Technical Trend", "technical_trend_score", .15),
    ("Event timing", "Event Timing", "event_timing_score", .10),
)


def _drivers(row, specs=LT_DRIVER_SPECS):
    """Explain a score as headline weight times distance from neutral (50)."""
    candidates = []
    for name, pillar, key, weight in specs:
        score = row.get(key)
        if pd.notna(score):
            contribution = weight * (score - 50)
            candidates.append({"driver_name":name,"pillar":pillar,"raw_value":score,
                               "component_score":score,"effective_weight":weight,
                               "contribution_points":contribution,
                               "direction":"POSITIVE" if contribution >= 0 else "NEGATIVE"})
    positive = [f"+{d['contribution_points']:.1f} pts — {d['driver_name']}" for d in sorted(candidates,key=lambda x:x['contribution_points'],reverse=True) if d["contribution_points"]>0][:3]
    negative = [f"{d['contribution_points']:.1f} pts — {d['driver_name']}" for d in sorted(candidates,key=lambda x:x['contribution_points']) if d["contribution_points"]<0][:3]
    return (
        "; ".join(positive) or "No dominant positive driver",
        "; ".join(negative) or "No dominant negative driver",
        candidates,
    )


def build_analysis(
    universe: pd.DataFrame, prices: pd.DataFrame, provider: pd.DataFrame
):
    frame = universe.merge(prices, on="symbol", how="left").merge(
        provider, on="symbol", how="left"
    )
    optional_columns = (
        "target_low",
        "target_mean",
        "target_median",
        "target_high",
        "current_price",
        "market_cap",
        "free_cash_flow",
        "forward_pe",
        "ev_ebitda",
        "peg",
        "price_book",
    )
    for column in optional_columns:
        if column not in frame:
            frame[column] = np.nan
    frame["sector_raw_constituent"] = frame.get("sector", frame.get("gics_sector"))
    yahoo = frame.get("sector_raw_yahoo", pd.Series(np.nan, index=frame.index))
    frame["sector_normalized"] = yahoo.combine_first(
        frame["sector_raw_constituent"]
    ).map(normalize_sector)
    frame["sector_source"] = np.where(yahoo.notna(), "YAHOO", "CONSTITUENT")
    frame["sector_normalization_method"] = "CANONICAL_MAP"
    frame["sector"] = frame["sector_normalized"]
    targets = target_metrics(frame)
    frame[["target_range_valid", "target_upside_pct", "target_dispersion_pct"]] = targets[
        ["target_range_valid", "target_upside_pct", "target_dispersion_pct"]
    ]
    provider_price = pd.to_numeric(frame.get("current_price_provider", pd.Series(np.nan, index=frame.index)), errors="coerce")
    completed_price = pd.to_numeric(frame.get("current_price", pd.Series(np.nan, index=frame.index)), errors="coerce")
    frame["provider_price_vs_last_completed_close_pct"] = np.where(
        provider_price.notna() & completed_price.gt(0),
        (provider_price / completed_price - 1) * 100, np.nan)
    frame["fcf_yield"] = np.where(
        frame.market_cap.gt(0), frame.free_cash_flow / frame.market_cap, np.nan
    )
    frame["forward_growth_scale_warning"] = (
        frame.get(
            "forward_eps_growth_scale_warning", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        | frame.get(
            "forward_revenue_growth_scale_warning",
            pd.Series(False, index=frame.index),
        ).fillna(False).astype(bool)
    )
    history_scores = frame.apply(
        lambda row: _history_scores_v31(row), axis=1, result_type="expand"
    )
    frame[["target_momentum_score", "revenue_revision_short_score", "revenue_revision_long_score"]] = history_scores
    frame["revenue_revision_momentum_score"] = frame["revenue_revision_long_score"]
    frame = add_peer_percentiles(
        frame,
        {
            "forward_pe": False,
            "ev_ebitda": False,
            "fcf_yield": True,
            "peg": False,
            "price_book": False,
        },
    )
    frame = add_relative_strength(frame)
    frame = add_benchmark_relative_strength(frame)
    frame = pd.concat(
        [
            frame.reset_index(drop=True),
            pd.DataFrame([expectations_score(row) for row in frame.to_dict("records")]),
        ],
        axis=1,
    )
    frame["market_price_risk_score"] = [market_price_risk_score(row) for row in frame.to_dict("records")]
    frame = pd.concat(
        [
            frame,
            pd.DataFrame([score_short_term(row) for row in frame.to_dict("records")]),
        ],
        axis=1,
    )
    frame = pd.concat(
        [
            frame,
            pd.DataFrame([score_long_term(row) for row in frame.to_dict("records")]),
        ],
        axis=1,
    )
    risk_results = pd.DataFrame(
        [risk_and_confidence(row) for row in frame.to_dict("records")],
        index=frame.index,
    )
    for column in risk_results:
        frame[column] = risk_results[column]
    long_term, short_term = rank_results(frame)
    frame["long_term_rank"] = frame.symbol.map(
        long_term.set_index("symbol").long_term_rank
    )
    frame["short_term_rank"] = frame.symbol.map(
        short_term.set_index("symbol").short_term_rank
    )
    lt_drivers = frame.apply(
        lambda row: _drivers(row, LT_DRIVER_SPECS), axis=1, result_type="expand"
    )
    frame[
        ["lt_positive_drivers", "lt_negative_drivers", "lt_driver_contributions"]
    ] = lt_drivers
    st_drivers = frame.apply(
        lambda row: _drivers(row, ST_DRIVER_SPECS), axis=1, result_type="expand"
    )
    frame[
        ["st_positive_drivers", "st_negative_drivers", "st_driver_contributions"]
    ] = st_drivers
    for horizon in ("lt", "st"):
        frame[f"{horizon}_top_positive_driver"] = (
            frame[f"{horizon}_positive_drivers"].str.split(";").str[0]
        )
        frame[f"{horizon}_top_negative_driver"] = (
            frame[f"{horizon}_negative_drivers"].str.split(";").str[0]
        )
    # Compatibility aliases remain LT-specific; reports use explicit horizon fields.
    frame["positive_drivers"] = frame["lt_positive_drivers"]
    frame["negative_drivers"] = frame["lt_negative_drivers"]
    frame["driver_contributions"] = frame["lt_driver_contributions"]
    frame["top_positive_driver"] = frame["lt_top_positive_driver"]
    frame["top_negative_driver"] = frame["lt_top_negative_driver"]
    frame["analyst_data_status"] = frame.get(
        "analyst_cache_status", pd.Series("INSUFFICIENT", index=frame.index)
    ).replace({FRESH_CACHE: FRESH, FRESH_PROVIDER: FRESH})
    frame["fundamental_data_status"] = frame.get(
        "fundamental_cache_status", pd.Series("INSUFFICIENT", index=frame.index)
    ).replace({FRESH_CACHE: FRESH, FRESH_PROVIDER: FRESH})
    frame["valuation_data_status"] = frame.get(
        "valuation_cache_status", pd.Series("INSUFFICIENT", index=frame.index)
    ).replace({FRESH_CACHE: FRESH, FRESH_PROVIDER: FRESH})
    frame["price_data_status"] = frame.get(
        "price_data_status", pd.Series("INSUFFICIENT", index=frame.index)
    ).fillna("INSUFFICIENT")
    status_columns = [
        "analyst_data_status",
        "fundamental_data_status",
        "valuation_data_status",
        "price_data_status",
    ]
    frame["overall_data_status"] = frame[status_columns].apply(
        lambda values: (
            ERROR
            if values.eq(ERROR).any()
            else (
                STALE_FALLBACK
                if values.isin([STALE_FALLBACK, STALE]).any()
                else PARTIAL if not values.eq(FRESH).all() else FRESH
            )
        ),
        axis=1,
    )
    return (
        frame,
        frame[frame.long_term_rank.notna()].sort_values("long_term_rank"),
        frame[frame.short_term_rank.notna()].sort_values("short_term_rank"),
    )

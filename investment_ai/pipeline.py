"""Merge provider inputs and calculate deterministic scores."""

from __future__ import annotations
import numpy as np
import pandas as pd
from investment_ai.features.analyst import expectations_score
from investment_ai.features.risk import risk_and_confidence
from investment_ai.features.technical import add_relative_strength
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


def _history_scores(row):
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
    revenue = {
        "0y": curve(
            row.get("revenue_0y_change_30d_pct"), [(-15, 0), (0, 50), (15, 100)]
        ),
        "plus_1y": curve(
            row.get("revenue_plus_1y_change_30d_pct"), [(-15, 0), (0, 50), (15, 100)]
        ),
        "0q": curve(
            row.get("revenue_0q_change_30d_pct"), [(-20, 0), (0, 50), (20, 100)]
        ),
        "plus_1q": curve(
            row.get("revenue_plus_1q_change_30d_pct"), [(-20, 0), (0, 50), (20, 100)]
        ),
    }
    revenue_score = weighted(
        revenue, {"0y": 0.35, "plus_1y": 0.40, "0q": 0.10, "plus_1q": 0.15}, 0
    )[0]
    return target_score, revenue_score


def _drivers(row):
    candidates = []
    for label, key, pillar in (
        ("EPS estimates", "eps_0q_change_30d_pct", "Expectations"),
        ("Revenue estimates", "revenue_0y_change_30d_pct", "Expectations"),
        ("Target median", "target_median_change_30d_pct", "Expectations"),
        ("Operating margin trend", "operating_margin_change_1y", "Quality"),
    ):
        value = row.get(key)
        if pd.notna(value):
            candidates.append(
                (abs(value), value >= 0, f"{label} {value:+.1f}%", pillar)
            )
    if pd.notna(row.get("target_dispersion_pct")) and row["target_dispersion_pct"] > 30:
        candidates.append(
            (
                row["target_dispersion_pct"],
                False,
                f"High target dispersion ({row['target_dispersion_pct']:.0f}%)",
                "Risk",
            )
        )
    if pd.notna(row.get("days_to_next_earnings")) and row["days_to_next_earnings"] <= 7:
        candidates.append(
            (50, False, f"Earnings in {row['days_to_next_earnings']:.0f} days", "Event")
        )
    positive = [
        text for _, direction, text, _ in sorted(candidates, reverse=True) if direction
    ][:3]
    negative = [
        text
        for _, direction, text, _ in sorted(candidates, reverse=True)
        if not direction
    ][:3]
    return (
        "; ".join(positive) or "No dominant positive driver",
        "; ".join(negative) or "No dominant negative driver",
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
    selected = (
        frame.get("target_median", pd.Series(np.nan, index=frame.index))
        .where(frame.get("target_median", pd.Series(np.nan, index=frame.index)) > 0)
        .combine_first(
            frame.get("target_mean", pd.Series(np.nan, index=frame.index)).where(
                frame.get("target_mean", pd.Series(np.nan, index=frame.index)) > 0
            )
        )
    )
    valid_range = frame.get("target_low", pd.Series(np.nan, index=frame.index)).le(
        frame.get("target_high", pd.Series(np.nan, index=frame.index))
    )
    frame["target_range_valid"] = valid_range
    frame["target_upside_pct"] = np.where(
        valid_range & selected.gt(0) & frame.current_price.gt(0),
        (selected / frame.current_price - 1) * 100,
        np.nan,
    )
    frame["target_dispersion_pct"] = np.where(
        valid_range & selected.gt(0),
        (frame.target_high - frame.target_low) / selected.abs() * 100,
        np.nan,
    )
    frame["fcf_yield"] = np.where(
        frame.market_cap.gt(0), frame.free_cash_flow / frame.market_cap, np.nan
    )
    history_scores = frame.apply(
        lambda row: _history_scores(row), axis=1, result_type="expand"
    )
    frame[["target_momentum_score", "revenue_revision_momentum_score"]] = history_scores
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
    frame = pd.concat(
        [
            frame.reset_index(drop=True),
            pd.DataFrame([expectations_score(row) for row in frame.to_dict("records")]),
        ],
        axis=1,
    )
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
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [risk_and_confidence(row) for row in frame.to_dict("records")]
            ),
        ],
        axis=1,
    )
    long_term, short_term = rank_results(frame)
    frame["long_term_rank"] = frame.symbol.map(
        long_term.set_index("symbol").long_term_rank
    )
    frame["short_term_rank"] = frame.symbol.map(
        short_term.set_index("symbol").short_term_rank
    )
    drivers = frame.apply(_drivers, axis=1, result_type="expand")
    frame[["positive_drivers", "negative_drivers"]] = drivers
    frame["top_positive_driver"] = frame.positive_drivers.str.split(";").str[0]
    frame["top_negative_driver"] = frame.negative_drivers.str.split(";").str[0]
    frame["analyst_data_status"] = frame.get(
        "analyst_cache_status", pd.Series("INSUFFICIENT", index=frame.index)
    ).replace({"cache": "FRESH", "provider": "FRESH"})
    frame["fundamental_data_status"] = frame.get(
        "fundamental_cache_status", pd.Series("INSUFFICIENT", index=frame.index)
    ).replace({"cache": "FRESH", "provider": "FRESH"})
    frame["valuation_data_status"] = frame.get(
        "valuation_cache_status", pd.Series("INSUFFICIENT", index=frame.index)
    ).replace({"cache": "FRESH", "provider": "FRESH"})
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
            "ERROR"
            if values.astype(str).str.contains("error", case=False).any()
            else (
                "STALE_FALLBACK"
                if values.astype(str).str.contains("stale", case=False).any()
                else "PARTIAL" if not values.eq("FRESH").all() else "FRESH"
            )
        ),
        axis=1,
    )
    return (
        frame,
        frame[frame.long_term_rank.notna()].sort_values("long_term_rank"),
        frame[frame.short_term_rank.notna()].sort_values("short_term_rank"),
    )

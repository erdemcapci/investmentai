from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


SCORING_MODEL_VERSION = "2.0"

RANKED = "RANKED"
PARTIAL_DATA = "PARTIAL_DATA"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

MIN_ANALYST_LONG_TERM_COVERAGE = 0.70
MIN_SHORT_TERM_COVERAGE = 0.75


def safe_float(value: Any) -> float:
    """Return a finite float or NaN for malformed, missing, or infinite input."""
    try:
        if value is None:
            return np.nan
        number = float(value)
        return number if math.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def piecewise_linear_score(value: Any, points: list[tuple[float, float]]) -> float:
    """Score a value using capped piecewise-linear interpolation."""
    number = safe_float(value)
    if pd.isna(number):
        return np.nan

    ordered = sorted(points)
    if number <= ordered[0][0]:
        return float(np.clip(ordered[0][1], 0, 100))
    if number >= ordered[-1][0]:
        return float(np.clip(ordered[-1][1], 0, 100))

    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x0 <= number <= x1:
            ratio = (number - x0) / (x1 - x0)
            return float(np.clip(y0 + ratio * (y1 - y0), 0, 100))

    return np.nan


def weighted_score_available(
    components: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, float]:
    """
    Return normalized score and available original-weight ratio.

    NaN components are excluded from both numerator and denominator.
    """
    total_weight = sum(weights.values())
    available_weight = 0.0
    weighted_sum = 0.0

    for name, weight in weights.items():
        value = safe_float(components.get(name))
        if pd.notna(value):
            available_weight += weight
            weighted_sum += value * weight

    if available_weight <= 0 or total_weight <= 0:
        return np.nan, 0.0

    return (
        float(np.clip(weighted_sum / available_weight, 0, 100)),
        float(available_weight / total_weight),
    )


def score_target_upside(upside_pct: Any) -> float:
    """Score analyst target upside with capped non-linear interpolation."""
    return piecewise_linear_score(
        upside_pct,
        [
            (-30, 0),
            (-10, 10),
            (0, 30),
            (10, 50),
            (25, 70),
            (40, 85),
            (60, 100),
        ],
    )


def score_analyst_sentiment(
    strong_buy: Any,
    buy: Any,
    hold: Any,
    sell: Any,
    strong_sell: Any,
) -> tuple[float, float, float, float, float]:
    """Return sentiment score, positive pct, negative pct, hold pct, strength score."""
    counts = [
        max(safe_float(strong_buy), 0),
        max(safe_float(buy), 0),
        max(safe_float(hold), 0),
        max(safe_float(sell), 0),
        max(safe_float(strong_sell), 0),
    ]
    counts = [0 if pd.isna(value) else value for value in counts]
    total = sum(counts)
    if total <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    positive_pct = (counts[0] + counts[1]) / total * 100
    negative_pct = (counts[3] + counts[4]) / total * 100
    hold_pct = counts[2] / total * 100
    strength = (
        counts[0] * 100
        + counts[1] * 80
        + counts[2] * 50
        + counts[3] * 20
        + counts[4] * 0
    ) / total
    sentiment = 0.70 * positive_pct + 0.30 * strength
    return (
        float(np.clip(sentiment, 0, 100)),
        float(positive_pct),
        float(negative_pct),
        float(hold_pct),
        float(strength),
    )


def score_target_agreement(dispersion_pct: Any) -> float:
    """Score analyst target agreement from target dispersion percentage."""
    return piecewise_linear_score(
        dispersion_pct,
        [
            (15, 100),
            (25, 85),
            (40, 65),
            (60, 35),
            (80, 0),
        ],
    )


def score_eps_revisions(upward_revisions: Any, downward_revisions: Any) -> tuple[float, bool]:
    """Score EPS revision balance; missing data returns NaN and unavailable."""
    up = safe_float(upward_revisions)
    down = safe_float(downward_revisions)
    if pd.isna(up) and pd.isna(down):
        return np.nan, False

    up = 0 if pd.isna(up) else max(up, 0)
    down = 0 if pd.isna(down) else max(down, 0)
    total = up + down
    if total == 0:
        return 50.0, True

    balance = (up - down) / max(total, 1)
    return float(np.clip((balance + 1) / 2 * 100, 0, 100)), True


def score_long_term_trend(price_vs_ma200_pct: Any) -> float:
    """Score long-term trend while penalizing severe downside and overextension."""
    return piecewise_linear_score(
        price_vs_ma200_pct,
        [
            (-30, 10),
            (-20, 25),
            (-10, 45),
            (0, 65),
            (10, 85),
            (20, 100),
            (35, 80),
            (50, 50),
        ],
    )


def calculate_coverage_confidence(rating_count: Any) -> float:
    """Map analyst count to a capped confidence factor."""
    return piecewise_linear_score(
        rating_count,
        [
            (5, 60),
            (10, 75),
            (15, 85),
            (20, 92),
            (30, 100),
        ],
    ) / 100


def calculate_coverage_multiplier(coverage_confidence: Any) -> float:
    """Soften analyst coverage confidence so moderate coverage does not dominate."""
    confidence = safe_float(coverage_confidence)
    if pd.isna(confidence):
        return np.nan
    return float(np.clip(0.70 + 0.30 * confidence, 0, 1))


def score_momentum(return_5d_pct: Any, return_20d_pct: Any) -> float:
    """Score momentum quality; healthy gains beat collapses and overextensions."""
    score_5d = piecewise_linear_score(
        return_5d_pct,
        [
            (-15, 0),
            (-10, 10),
            (-5, 35),
            (0, 60),
            (3, 85),
            (6, 100),
            (10, 80),
            (15, 50),
            (25, 20),
        ],
    )
    score_20d = piecewise_linear_score(
        return_20d_pct,
        [
            (-25, 0),
            (-15, 20),
            (-5, 45),
            (0, 60),
            (8, 85),
            (15, 100),
            (25, 75),
            (40, 40),
            (60, 20),
        ],
    )
    score, _ = weighted_score_available(
        {"score_5d": score_5d, "score_20d": score_20d},
        {"score_5d": 0.65, "score_20d": 0.35},
    )
    return score


def score_ma_distance(distance_pct: Any) -> float:
    """Score price distance from a moving average."""
    return piecewise_linear_score(
        distance_pct,
        [
            (-15, 10),
            (-10, 25),
            (-5, 45),
            (0, 70),
            (3, 90),
            (6, 100),
            (10, 80),
            (15, 55),
            (25, 25),
        ],
    )


def score_ma_alignment(price_vs_ma20_pct: Any, price_vs_ma50_pct: Any) -> float:
    """Score moving-average alignment from MA20 and MA50 distances."""
    score, _ = weighted_score_available(
        {
            "ma20": score_ma_distance(price_vs_ma20_pct),
            "ma50": score_ma_distance(price_vs_ma50_pct),
        },
        {"ma20": 0.60, "ma50": 0.40},
    )
    return score


def score_selloff_stability(
    return_2d_pct: Any,
    negative_days_last_5: Any,
    worst_daily_return_5d_pct: Any,
) -> tuple[float, float, str]:
    """Score recent instability using available subcomponents only."""
    two_day = piecewise_linear_score(
        return_2d_pct,
        [(-10, 0), (-6, 25), (-3, 55), (-1, 80), (1, 100), (4, 85)],
    )
    red_days = piecewise_linear_score(
        negative_days_last_5,
        [(0, 100), (1, 85), (2, 65), (3, 40), (4, 20), (5, 0)],
    )
    worst_day = piecewise_linear_score(
        worst_daily_return_5d_pct,
        [(-10, 0), (-6, 30), (-4, 55), (-2, 80), (-1, 100)],
    )
    score, coverage = weighted_score_available(
        {"two_day": two_day, "red_days": red_days, "worst_day": worst_day},
        {"two_day": 0.45, "red_days": 0.30, "worst_day": 0.25},
    )
    if pd.isna(score):
        status = INSUFFICIENT_DATA
    elif coverage < 1:
        status = PARTIAL_DATA
    else:
        status = RANKED
    return score, coverage * 100, status


def score_pullback_quality(drawdown_from_20d_high_pct: Any) -> float:
    """Score pullback quality with controlled pullbacks preferred."""
    return piecewise_linear_score(
        drawdown_from_20d_high_pct,
        [
            (-35, 0),
            (-25, 15),
            (-18, 45),
            (-12, 75),
            (-8, 95),
            (-6, 100),
            (-4, 90),
            (-2, 70),
            (0, 55),
        ],
    )


def score_absolute_volatility(volatility_pct: Any) -> float:
    """Fallback absolute volatility score."""
    return piecewise_linear_score(
        volatility_pct,
        [(20, 100), (35, 80), (50, 55), (65, 30), (80, 5)],
    )


def score_sector_volatility_percentile(percentile: Any) -> float:
    """Score sector-relative volatility percentile; lower is better."""
    return piecewise_linear_score(
        percentile,
        [(0, 100), (20, 90), (40, 75), (60, 55), (80, 30), (100, 5)],
    )


def score_earnings_timing(
    days_to_next_earnings: Any,
    days_since_last_earnings: Any = np.nan,
) -> float:
    """Score entry timing around earnings when date data is available."""
    next_days = safe_float(days_to_next_earnings)
    last_days = safe_float(days_since_last_earnings)

    if pd.notna(next_days) and next_days >= 0:
        return piecewise_linear_score(
            next_days,
            [(0, 10), (3, 10), (7, 30), (14, 55), (21, 70), (45, 90), (46, 100)],
        )

    if pd.notna(last_days) and last_days >= 0:
        return piecewise_linear_score(
            last_days,
            [(0, 35), (1, 35), (5, 60), (15, 85), (45, 95)],
        )

    return np.nan


def assign_long_term_category(score: Any) -> str:
    """Assign analyst-based long-term category."""
    value = safe_float(score)
    if pd.isna(value):
        return "INSUFFICIENT DATA"
    if value < 45:
        return "WEAK"
    if value < 60:
        return "NEUTRAL"
    if value < 75:
        return "POSITIVE"
    return "STRONG"


def assign_short_term_category(score: Any) -> str:
    """Assign short-term entry category."""
    value = safe_float(score)
    if pd.isna(value):
        return "INSUFFICIENT DATA"
    if value < 40:
        return "WAIT / AVOID"
    if value < 60:
        return "NEUTRAL"
    if value < 75:
        return "WATCH"
    return "FAVORABLE ENTRY"


def assign_candidate_profile(long_term_score: Any, short_term_score: Any) -> tuple[str, str]:
    """Assign a two-dimensional candidate profile and explanation."""
    lt = safe_float(long_term_score)
    st = safe_float(short_term_score)
    if pd.isna(lt) and pd.isna(st):
        return "INSUFFICIENT DATA", "Neither score has enough reliable data."
    if pd.isna(lt):
        return "TECHNICAL ONLY", "Entry score is available, but analyst-based score is insufficient."
    if pd.isna(st):
        return "ANALYST VIEW ONLY", "Analyst-based score is available, but entry timing data is insufficient."
    if lt >= 75 and st >= 75:
        return "STRONG CANDIDATE", "Strong analyst-based attractiveness and favorable entry timing."
    if lt >= 75 and st < 40:
        return "ATTRACTIVE BUT TECHNICALLY WEAK", "Strong analyst-based attractiveness, but the current technical setup is severely weak."
    if lt >= 75 and st < 60:
        return "ATTRACTIVE, WAIT FOR ENTRY", "Strong analyst-based attractiveness, but short-term timing is not yet favorable."
    if lt >= 60 and st >= 75:
        return "TACTICAL CANDIDATE", "Good analyst-based score with especially favorable entry timing."
    if lt < 60 and st >= 75:
        return "MOMENTUM ONLY", "Entry timing is strong, but analyst-based support is weaker."
    if lt >= 60 and st >= 60:
        return "RESEARCH CANDIDATE", "Both scores clear the research threshold."
    return "LOW PRIORITY", "Both scores do not clear the research thresholds."


def candidate_profile_priority(profile: Any) -> int:
    """Return sorting priority for profiles eligible for combined candidates."""
    return {
        "STRONG CANDIDATE": 1,
        "TACTICAL CANDIDATE": 2,
        "RESEARCH CANDIDATE": 3,
    }.get(str(profile), 99)


def format_driver_text(
    row: pd.Series,
) -> tuple[str, str]:
    """Create readable positive and negative driver summaries from scores and flags."""
    positive: list[str] = []
    negative: list[str] = []

    score_labels = {
        "lt_score_upside": "analyst upside",
        "lt_score_sentiment": "analyst sentiment",
        "lt_score_eps_revisions": "EPS revisions",
        "lt_score_target_agreement": "target agreement",
        "lt_score_long_trend": "long-term trend",
        "st_score_momentum": "momentum quality",
        "st_score_ma_alignment": "moving-average alignment",
        "st_score_pullback_quality": "pullback quality",
        "st_score_selloff_stability": "selloff stability",
        "st_score_volatility": "volatility profile",
        "st_score_earnings_timing": "earnings timing",
    }

    for column, label in score_labels.items():
        value = safe_float(row.get(column))
        if pd.isna(value):
            continue
        if value >= 75:
            positive.append(f"Strong {label}")
        elif value <= 35:
            negative.append(f"Weak {label}")

    flags = set(str(row.get("risk_flags", "")).split("|"))
    flag_negative = {
        "HIGH_TARGET_DISAGREEMENT": "High target disagreement",
        "EXTREME_TARGET_UPSIDE": "Extreme target upside may be stale or uncertain",
        "NEGATIVE_EPS_REVISIONS": "Negative EPS revisions",
        "EARNINGS_WITHIN_3_DAYS": "Earnings within 3 days",
        "EARNINGS_WITHIN_7_DAYS": "Earnings within 7 days",
        "POST_EARNINGS_PRICE_DISCOVERY": "Post-earnings price discovery",
        "SHARP_RECENT_SELLOFF": "Sharp recent selloff",
        "SHORT_TERM_OVEREXTENDED": "Short-term overextension",
        "EXTREME_OVEREXTENSION": "Extreme overextension",
        "LOW_LIQUIDITY": "Low liquidity warning",
        "VERY_LOW_LIQUIDITY": "Very low liquidity",
        "HIGH_VOLATILITY": "High volatility",
    }
    for flag, text in flag_negative.items():
        if flag in flags:
            negative.append(text)

    return (
        "; ".join(dict.fromkeys(positive)) or "No standout positive drivers",
        "; ".join(dict.fromkeys(negative)) or "No standout negative drivers",
    )

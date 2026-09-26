"""Parsers for yfinance analyst endpoints and Expectations scoring."""

from __future__ import annotations
import re
from typing import Any
import numpy as np
import pandas as pd
from investment_ai.scoring.common import (
    change_pct,
    curve,
    number,
    ratio,
    safe_nanmean,
    weighted,
)

PERIODS = ("0q", "+1q", "0y", "+1y")


def _frame(value: Any) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9+-]", "", str(value).lower())


def _row(frame: pd.DataFrame, label: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    labels = {_norm(index): index for index in frame.index}
    key = labels.get(_norm(label))
    return frame.loc[key] if key is not None else pd.Series(dtype=object)


def _get(row: pd.Series, *names: str) -> float:
    values = {_norm(key): value for key, value in row.items()}
    return next(
        (number(values[_norm(name)]) for name in names if _norm(name) in values), np.nan
    )


def parse_recommendations(value: Any) -> dict[str, Any]:
    """Parse both period-column and period-index recommendation summaries."""
    working = _frame(value)
    period_column = next(
        (column for column in working if _norm(column) == "period"), None
    )
    if period_column is not None:
        working = working.set_index(period_column)
    aliases = {
        "current": ("0m", "current"),
        "minus_1m": ("-1m",),
        "minus_2m": ("-2m",),
        "minus_3m": ("-3m",),
    }
    output: dict[str, Any] = {}
    for prefix, periods in aliases.items():
        row = next(
            (
                _row(working, period)
                for period in periods
                if not _row(working, period).empty
            ),
            pd.Series(dtype=object),
        )
        counts = [
            _get(row, name)
            for name in ("strongBuy", "buy", "hold", "sell", "strongSell")
        ]
        valid = all(pd.notna(item) and item >= 0 for item in counts)
        total = sum(counts) if valid else np.nan
        valid = valid and total > 0
        output[f"ratings_valid_{prefix}"] = bool(valid)
        output[f"rating_count_{prefix}"] = total if valid else np.nan
        output[f"positive_rating_pct_{prefix}"] = (
            (counts[0] + counts[1]) / total * 100 if valid else np.nan
        )
        output[f"negative_rating_pct_{prefix}"] = (
            (counts[3] + counts[4]) / total * 100 if valid else np.nan
        )
        output[f"hold_pct_{prefix}"] = counts[2] / total * 100 if valid else np.nan
        output[f"recommendation_strength_{prefix}"] = (
            sum(value * weight for value, weight in zip(counts, (100, 80, 50, 20, 0)))
            / total
            if valid
            else np.nan
        )
        if prefix == "current":
            output.update(
                dict(
                    zip(
                        ("strong_buy", "buy", "hold", "sell", "strong_sell"),
                        counts if valid else [np.nan] * 5,
                    )
                )
            )
    output["positive_rating_pct"] = output["positive_rating_pct_current"]
    output["rating_count"] = output["rating_count_current"]
    for months in (1, 3):
        suffix = f"minus_{months}m"
        output[f"positive_rating_change_{months}m_pp"] = number(
            output["positive_rating_pct_current"]
        ) - number(output[f"positive_rating_pct_{suffix}"])
        output[f"recommendation_strength_change_{months}m"] = number(
            output["recommendation_strength_current"]
        ) - number(output[f"recommendation_strength_{suffix}"])
    return output


def parse_eps_trend(value: Any) -> dict[str, Any]:
    output = {}
    frame = _frame(value)
    for period in PERIODS:
        row, prefix = _row(frame, period), period.replace("+", "plus_")
        current = _get(row, "current")
        output[f"eps_{prefix}_current"] = current
        for days in (7, 30, 60, 90):
            output[f"eps_{prefix}_change_{days}d_pct"] = change_pct(
                current, _get(row, f"{days}daysAgo")
            )
    return output


def parse_revisions(value: Any) -> dict[str, Any]:
    output, frame = {}, _frame(value)
    for period in PERIODS:
        row, prefix = _row(frame, period), period.replace("+", "plus_")
        for days in (7, 30):
            up, down = _get(row, f"upLast{days}days"), _get(row, f"downLast{days}days")
            total = up + down
            output[f"eps_revision_breadth_{prefix}_{days}d"] = (
                (up - down) / total if pd.notna(total) and total > 0 else np.nan
            )
            output[f"eps_up_{prefix}_{days}d"], output[f"eps_down_{prefix}_{days}d"] = (
                up,
                down,
            )
    return output


def parse_estimates(value: Any, kind: str) -> dict[str, Any]:
    output, frame = {}, _frame(value)
    names = (
        "numberOfAnalysts",
        "avg",
        "low",
        "high",
        "yearAgoEps" if kind == "eps" else "yearAgoRevenue",
        "growth",
    )
    for period in PERIODS:
        row, prefix = _row(frame, period), period.replace("+", "plus_")
        for name in names:
            output[f"{kind}_{prefix}_{name}"] = _get(row, name)
        avg = output[f"{kind}_{prefix}_avg"]
        output[f"{kind}_{prefix}_dispersion_pct"] = (
            ratio(
                output[f"{kind}_{prefix}_high"] - output[f"{kind}_{prefix}_low"],
                abs(avg),
            )
            * 100
        )
    plus_1y = number(output.get(f"{kind}_plus_1y_growth"))
    current_year = number(output.get(f"{kind}_0y_growth"))
    plus_valid = pd.notna(plus_1y) and abs(plus_1y) <= 5
    current_valid = pd.notna(current_year) and abs(current_year) <= 5
    growth = plus_1y if plus_valid else current_year if current_valid else np.nan
    output[f"forward_{kind}_growth"] = growth
    output[f"forward_{kind}_growth_source"] = (
        "+1y" if plus_valid else "0y" if current_valid else "unavailable"
    )
    output[f"forward_{kind}_growth_scale_warning"] = bool(
        (pd.notna(plus_1y) and not plus_valid)
        or (pd.notna(current_year) and not current_valid)
    )
    return output


def parse_targets(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        source = value.iloc[:, 0].to_dict() if not value.empty else {}
    else:
        source = value if isinstance(value, dict) else {}
    normalized = {_norm(key): val for key, val in source.items()}

    def get(*keys: str) -> float:
        return next(
            (
                number(normalized[_norm(key)])
                for key in keys
                if _norm(key) in normalized
            ),
            np.nan,
        )

    return {
        "target_low": get("low", "targetLowPrice"),
        "target_mean": get("mean", "targetMeanPrice"),
        "target_median": get("median", "targetMedianPrice"),
        "target_high": get("high", "targetHighPrice"),
    }


def parse_actions(value: Any, now: pd.Timestamp | None = None) -> dict[str, Any]:
    frame = _frame(value)
    if frame.empty:
        return {}
    frame.columns = [_norm(column) for column in frame.columns]
    now = now or pd.Timestamp.now(tz="UTC")
    date_column = next(
        (column for column in ("gradedate", "date") if column in frame), None
    )
    dates = pd.to_datetime(
        frame[date_column] if date_column else frame.index, utc=True, errors="coerce"
    )
    frame = frame.assign(_date=np.asarray(dates)).sort_values("_date", ascending=False)
    dates = pd.DatetimeIndex(frame["_date"])
    actions = (
        frame.get("action", pd.Series("", index=frame.index)).astype(str).str.lower()
    )
    output = {}
    for days in (7, 30, 90):
        selected = actions[dates >= now - pd.Timedelta(days=days)]
        upgrades, downgrades = (
            selected.str.contains("up").sum(),
            selected.str.contains("down").sum(),
        )
        output.update(
            {
                f"rating_actions_{days}d": len(selected),
                f"upgrades_{days}d": upgrades,
                f"downgrades_{days}d": downgrades,
                f"rating_action_balance_{days}d": (
                    (upgrades - downgrades) / (upgrades + downgrades)
                    if upgrades + downgrades
                    else np.nan
                ),
            }
        )
    last = frame.iloc[0]
    output.update(
        {
            "latest_rating_date": str(last["_date"]),
            "latest_rating_firm": last.get("firm", ""),
            "latest_rating_action": last.get("action", ""),
            "latest_rating_from": last.get("fromgrade", ""),
            "latest_rating_to": last.get("tograde", ""),
        }
    )
    return output


def parse_surprises(value: Any) -> dict[str, Any]:
    frame = _frame(value)
    column = next(
        (
            column
            for column in frame
            if _norm(column) in {"surprisepercent", "surprise%", "surprisepct"}
        ),
        None,
    )
    if not frame.empty:
        dates = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.assign(_observation_date=np.asarray(dates)).sort_values(
            "_observation_date"
        )
    values = (
        pd.to_numeric(frame[column], errors="coerce").dropna().tail(4)
        if column
        else pd.Series(dtype=float)
    )
    if len(values) < 2:
        return {}
    if values.abs().max() <= 2:
        values *= 100
    return {
        "positive_surprise_rate_4q": (values > 0).mean() * 100,
        "median_eps_surprise_4q": values.median(),
        "mean_eps_surprise_4q": values.mean(),
        "eps_surprise_volatility_4q": values.std(ddof=0),
        "earnings_surprise_observation_count": len(values),
    }


def momentum_score(
    row: dict[str, Any],
    prefix: str,
    weights: dict[str, float],
    window_weights: dict[int, float] | None = None,
) -> float:
    window_weights = window_weights or {7: 0.25, 30: 0.25, 60: 0.25, 90: 0.25}
    components = {
        period: weighted(
            {
                str(days): curve(
                    row.get(f"{prefix}_{period}_change_{days}d_pct"),
                    [(-20, 0), (0, 50), (20, 100)],
                )
                for days in window_weights
            },
            {str(days): weight for days, weight in window_weights.items()},
            0,
        )[0]
        for period in weights
    }
    return weighted(components, weights, 0)[0]


def expectations_score(row: dict[str, Any]) -> dict[str, Any]:
    short_windows = {7: 0.50, 30: 0.30, 60: 0.15, 90: 0.05}
    long_windows = {7: 0.30, 30: 0.30, 60: 0.20, 90: 0.20}
    short_eps = momentum_score(row, "eps", {"0q": 0.55, "plus_1q": 0.45}, short_windows)
    long_eps = momentum_score(
        row,
        "eps",
        {"0q": 0.15, "plus_1q": 0.20, "0y": 0.30, "plus_1y": 0.35},
        long_windows,
    )
    long_breadth = curve(
        safe_nanmean(
            [
                row.get(f"eps_revision_breadth_{period}_{days}d", np.nan)
                for period in ("0q", "plus_1q", "0y", "plus_1y")
                for days in (7, 30)
            ]
        ),
        [(-1, 0), (0, 50), (1, 100)],
    )
    short_breadth = curve(
        weighted(
            {
                f"{p}_{d}": row.get(f"eps_revision_breadth_{p}_{d}d")
                for p in ("0q", "plus_1q")
                for d in (7, 30)
            },
            {"0q_7": 0.35, "0q_30": 0.20, "plus_1q_7": 0.25, "plus_1q_30": 0.20},
            0,
        )[0],
        [(-1, 0), (0, 50), (1, 100)],
    )
    recommendation = curve(
        safe_nanmean(
            [
                row.get("positive_rating_change_1m_pp", np.nan),
                row.get("positive_rating_change_3m_pp", np.nan),
            ]
        ),
        [(-20, 0), (0, 50), (20, 100)],
    )
    actions = curve(row.get("rating_action_balance_30d"), [(-1, 0), (0, 50), (1, 100)])
    target = curve(
        row.get("target_upside_pct"), [(-30, 0), (0, 40), (30, 80), (60, 100)]
    )
    execution = curve(
        row.get("positive_surprise_rate_4q"), [(0, 0), (50, 50), (100, 100)]
    )
    consistency = curve(
        row.get("eps_surprise_volatility_4q"), [(0, 100), (10, 70), (30, 20), (60, 0)]
    )
    period_keys = tuple(period.replace("+", "plus_") for period in PERIODS)
    recent = weighted(
        {
            "7": safe_nanmean([row.get(f"eps_{p}_change_7d_pct") for p in period_keys]),
            "30": safe_nanmean(
                [row.get(f"eps_{p}_change_30d_pct") for p in period_keys]
            ),
        },
        {"7": 0.625, "30": 0.375},
        0,
    )[0]
    older = weighted(
        {
            "60": safe_nanmean(
                [row.get(f"eps_{p}_change_60d_pct") for p in period_keys]
            ),
            "90": safe_nanmean(
                [row.get(f"eps_{p}_change_90d_pct") for p in period_keys]
            ),
        },
        {"60": 0.5, "90": 0.5},
        0,
    )[0]
    acceleration = recent - older if pd.notna(recent) and pd.notna(older) else np.nan
    long_components = {
        "eps_momentum": long_eps,
        "revision_breadth": long_breadth,
        "revenue_revision_momentum": row.get("revenue_revision_long_score"),
        "recommendation_trend": recommendation,
        "rating_actions": actions,
        "target_momentum": row.get("target_momentum_score"),
        "target_signal": target,
        "earnings_execution": execution,
        "expectation_consistency": consistency,
    }
    long_weights = {
        "eps_momentum": 0.25,
        "revision_breadth": 0.15,
        "revenue_revision_momentum": 0.15,
        "recommendation_trend": 0.10,
        "rating_actions": 0.10,
        "target_momentum": 0.10,
        "target_signal": 0.05,
        "earnings_execution": 0.05,
        "expectation_consistency": 0.05,
    }
    short_components = {
        "eps_momentum": short_eps,
        "revision_breadth": short_breadth,
        "revenue_revision_momentum": row.get("revenue_revision_short_score"),
        "recommendation_trend": curve(
            row.get("positive_rating_change_1m_pp"), [(-20, 0), (0, 50), (20, 100)]
        ),
        "rating_actions": curve(
            row.get("rating_action_balance_7d"), [(-1, 0), (0, 50), (1, 100)]
        ),
        "target_momentum": row.get("target_momentum_score"),
        "earnings_execution": execution,
    }
    short_weights = {
        "eps_momentum": 0.35,
        "revision_breadth": 0.20,
        "revenue_revision_momentum": 0.15,
        "recommendation_trend": 0.10,
        "rating_actions": 0.10,
        "target_momentum": 0.05,
        "earnings_execution": 0.05,
    }
    long_score, long_coverage, long_status = weighted(
        long_components, long_weights, 0.60
    )
    short_score, short_coverage, short_status = weighted(
        short_components, short_weights, 0.60
    )
    directional = {
        "eps": curve(recent, [(-20, 0), (0, 50), (20, 100)]),
        "breadth": short_breadth,
        "revenue": safe_nanmean(
            [
                row.get("revenue_revision_short_score"),
                row.get("revenue_revision_long_score"),
            ]
        ),
        "recommendations": recommendation,
        "actions": actions,
        "targets": row.get("target_momentum_score"),
    }
    direction_score, direction_coverage, _ = weighted(
        directional,
        {
            "eps": 0.30,
            "breadth": 0.20,
            "revenue": 0.20,
            "recommendations": 0.10,
            "actions": 0.10,
            "targets": 0.10,
        },
        0.50,
    )
    trend = (
        "INSUFFICIENT_DATA"
        if pd.isna(direction_score)
        else (
            "STRONGLY_IMPROVING"
            if direction_score >= 80
            else (
                "IMPROVING"
                if direction_score >= 60
                else (
                    "STABLE"
                    if direction_score >= 40
                    else "DETERIORATING"
                    if direction_score >= 20
                    else "RAPIDLY_DETERIORATING"
                )
            )
        )
    )
    return {
        "eps_short_horizon_momentum_score": short_eps,
        "eps_long_horizon_momentum_score": long_eps,
        "eps_revision_acceleration": acceleration,
        "eps_revision_acceleration_score": curve(
            acceleration, [(-20, 0), (0, 50), (20, 100)]
        ),
        "expectations_long_score": long_score,
        "expectations_long_coverage": long_coverage,
        "expectations_long_status": long_status,
        "expectations_short_score": short_score,
        "expectations_short_coverage": short_coverage,
        "expectations_short_status": short_status,
        "expectations_direction_score": direction_score,
        "expectations_direction_coverage": direction_coverage,
        "expectations_direction": trend,
        # Compatibility alias; primary models never consume it.
        "expectations_score": long_score,
        "expectations_coverage": long_coverage,
        "expectations_status": long_status,
        "expectations_trend": trend,
    }

"""Parsers for yfinance analyst endpoints and Expectations scoring."""

from __future__ import annotations
import re
from typing import Any
import numpy as np
import pandas as pd
from investment_ai.scoring.common import change_pct, curve, number, ratio, weighted

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
    output[f"forward_{kind}_growth"] = output.get(
        f"{kind}_plus_1y_growth", output.get(f"{kind}_0y_growth", np.nan)
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
    }


def momentum_score(
    row: dict[str, Any], prefix: str, weights: dict[str, float]
) -> float:
    components = {
        period: np.nanmean(
            [
                curve(
                    row.get(f"{prefix}_{period}_change_{days}d_pct"),
                    [(-20, 0), (0, 50), (20, 100)],
                )
                for days in (7, 30, 60, 90)
            ]
        )
        for period in weights
    }
    return weighted(components, weights, 0)[0]


def expectations_score(row: dict[str, Any]) -> dict[str, Any]:
    short_eps = momentum_score(row, "eps", {"0q": 0.55, "plus_1q": 0.45})
    long_eps = momentum_score(
        row, "eps", {"0q": 0.15, "plus_1q": 0.20, "0y": 0.30, "plus_1y": 0.35}
    )
    breadth = curve(
        np.nanmean(
            [
                row.get(f"eps_revision_breadth_{period}_{days}d", np.nan)
                for period in ("0q", "plus_1q", "0y", "plus_1y")
                for days in (7, 30)
            ]
        ),
        [(-1, 0), (0, 50), (1, 100)],
    )
    recommendation = curve(
        np.nanmean(
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
    components = {
        "eps_momentum": long_eps,
        "revision_breadth": breadth,
        "revenue_revision_momentum": row.get("revenue_revision_momentum_score"),
        "recommendation_trend": recommendation,
        "rating_actions": actions,
        "target_momentum": row.get("target_momentum_score"),
        "target_signal": target,
        "earnings_execution": execution,
        "expectation_consistency": consistency,
    }
    weights = {
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
    score, coverage, status = weighted(components, weights, 0.60)
    trend = (
        "INSUFFICIENT_DATA"
        if pd.isna(score)
        else (
            "STRONGLY_IMPROVING"
            if score >= 80
            else (
                "IMPROVING"
                if score >= 60
                else (
                    "STABLE"
                    if score >= 40
                    else "DETERIORATING" if score >= 20 else "RAPIDLY_DETERIORATING"
                )
            )
        )
    )
    return {
        **{f"expectations_{key}_score": value for key, value in components.items()},
        "eps_short_horizon_momentum_score": short_eps,
        "eps_long_horizon_momentum_score": long_eps,
        "expectations_score": score,
        "expectations_coverage": coverage,
        "expectations_status": status,
        "expectations_trend": trend,
    }

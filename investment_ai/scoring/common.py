from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

RANKED = "RANKED"
PARTIAL_DATA = "PARTIAL_DATA"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
HISTORY_NOT_YET_AVAILABLE = "HISTORY_NOT_YET_AVAILABLE"


def number(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def safe_nanmean(values: Any) -> float:
    """Mean of finite numeric values without NumPy's all-NaN warning."""
    valid = [number(value) for value in values]
    valid = [value for value in valid if pd.notna(value)]
    return float(np.mean(valid)) if valid else np.nan


def ratio(a: Any, b: Any) -> float:
    a, b = number(a), number(b)
    return np.nan if pd.isna(a) or pd.isna(b) or b == 0 else a / b


def change_pct(current: Any, previous: Any) -> float:
    """Percentage change using abs(previous); negative EPS is handled correctly."""
    c, p = number(current), number(previous)
    return np.nan if pd.isna(c) or pd.isna(p) or p == 0 else (c - p) / abs(p) * 100


def curve(value: Any, points: list[tuple[float, float]]) -> float:
    x = number(value)
    if pd.isna(x):
        return np.nan
    p = sorted(points)
    return float(np.clip(np.interp(x, [a for a, _ in p], [b for _, b in p]), 0, 100))


NEUTRAL = 50.0


def weighted(
    components: dict[str, Any],
    weights: dict[str, float],
    minimum: float = 0,
    neutral_fill: float | None = None,
) -> tuple[float, float, str]:
    """Weighted mean of available components.

    By default missing components are dropped and the remaining weights are
    renormalised.  With ``neutral_fill`` a missing component instead counts as
    that value, so absent evidence shrinks the score toward neutral rather than
    letting the available components speak for the missing weight.  Coverage
    always reflects only the genuinely available weight.
    """
    total = sum(weights.values())
    available = [
        (number(components.get(k)), w)
        for k, w in weights.items()
        if pd.notna(number(components.get(k)))
    ]
    coverage = sum(w for _, w in available) / total if total else 0
    if not available or coverage < minimum:
        return np.nan, coverage, INSUFFICIENT_DATA
    if neutral_fill is not None:
        filled = available + [(neutral_fill, total * (1 - coverage))]
        score = sum(v * w for v, w in filled) / total
    else:
        score = sum(v * w for v, w in available) / sum(w for _, w in available)
    score = np.clip(score, 0, 100)
    return float(score), float(coverage), RANKED if coverage == 1 else PARTIAL_DATA


def percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    count = numeric.notna().sum()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if count == 1:
        result.loc[numeric.notna()] = 50.0
    elif count > 1:
        ranks = numeric.rank(method="average")
        result = (ranks - 1) / (count - 1) * 100
    return result if higher_is_better else 100 - result

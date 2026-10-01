"""Auditable regional benchmark-relative return features (experimental in v1.0)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from investment_ai.features.peers import fallback_percentile, memberships
from investment_ai.features.technical import RS_METRICS

RISK_ON = "RISK_ON"
RISK_OFF = "RISK_OFF"
REBOUND_RISK = "REBOUND_RISK"
REGIME_UNKNOWN = "UNKNOWN"

BENCHMARKS = {
    "S&P 500": ("^GSPC", "S&P 500 Index", "Yahoo direct index"),
    "STOXX Europe 600": ("^STOXX", "STOXX Europe 600 Index", "Yahoo direct index"),
}


def attach_benchmark_prices(universe: pd.DataFrame, prices: pd.DataFrame,
                            benchmark_prices: pd.DataFrame) -> pd.DataFrame:
    """Attach only the assigned region's benchmark observations to each security."""
    result = prices.copy()
    index_by_symbol = benchmark_prices.set_index("symbol") if not benchmark_prices.empty else pd.DataFrame()
    assignments = universe.set_index("symbol").index_name.map(assign_benchmark)
    assignment_map = dict(zip(universe.symbol, assignments))
    for metric in RS_METRICS.values():
        result[f"benchmark_{metric}"] = result.symbol.map(
            lambda symbol: index_by_symbol.at[assignment_map[symbol][0], metric]
            if assignment_map.get(symbol, ("",))[0] in index_by_symbol.index
            and metric in index_by_symbol else np.nan
        )
    for source, target in (("current_price", "benchmark_price"), ("price_as_of", "benchmark_price_as_of"),
                           ("price_data_status", "benchmark_price_status"),
                           ("price_vs_ma200_pct", "benchmark_price_vs_ma200_pct")):
        result[target] = result.symbol.map(
            lambda symbol: index_by_symbol.at[assignment_map[symbol][0], source]
            if assignment_map.get(symbol, ("",))[0] in index_by_symbol.index and source in index_by_symbol else np.nan
        )
    return result


def assign_benchmark(index_name: object) -> tuple[str, str, str]:
    """Pick a deterministic primary membership; lexical tie-break matches peer logic."""
    eligible = sorted(set(memberships(index_name)) & set(BENCHMARKS))
    return BENCHMARKS[eligible[0]] if eligible else ("", "", "UNASSIGNED")


def add_benchmark_relative_strength(frame: pd.DataFrame, minimum_peers: int = 15) -> pd.DataFrame:
    result = frame.copy()
    assigned = result.get("index_name", pd.Series("", index=result.index)).map(assign_benchmark)
    result[["benchmark_symbol", "benchmark_name", "benchmark_assignment_method"]] = pd.DataFrame(
        assigned.tolist(), index=result.index
    )
    result["benchmark_return_basis"] = "ADJUSTED_CLOSE_RETURN"
    sector = result.get("sector_normalized", result.get("sector", pd.Series("", index=result.index)))
    excess_columns = []
    for key, metric in RS_METRICS.items():
        stock = pd.to_numeric(result.get(metric), errors="coerce")
        benchmark = pd.to_numeric(result.get(f"benchmark_{metric}"), errors="coerce")
        column = f"excess_{metric}"
        result[column] = stock - benchmark
        excess_columns.append(column)
        # Sector first, then the assigned benchmark region, then the universe.
        result[f"benchmark_rs_{key}_percentile"] = fallback_percentile(
            result[column], [sector, result.benchmark_symbol], minimum_peers
        )
    available = result[excess_columns].notna().any(axis=1)
    result["benchmark_rs_status"] = np.where(available, "AVAILABLE", "FALLBACK_RAW")
    return result


def market_regime(benchmark_vs_ma200: object, benchmark_return_20d: object) -> str:
    """Classify the assigned market's trend for momentum-crash protection.

    Momentum strategies suffer their worst losses when a falling market
    rebounds sharply, so a strong 20-day rally below the 200-day average is
    flagged separately from an ordinary downtrend.
    """
    ma = pd.to_numeric(benchmark_vs_ma200, errors="coerce")
    ret = pd.to_numeric(benchmark_return_20d, errors="coerce")
    if pd.isna(ma):
        return REGIME_UNKNOWN
    if ma >= 0:
        return RISK_ON
    if pd.notna(ret) and ret >= 5:
        return REBOUND_RISK
    return RISK_OFF

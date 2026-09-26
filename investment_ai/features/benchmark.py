"""Auditable regional benchmark-relative return features (experimental in v1.0)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from investment_ai.features.peers import memberships
from investment_ai.scoring.common import percentile

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
    for horizon in (20, 60, 126, 252):
        result[f"benchmark_return_{horizon}d_pct"] = result.symbol.map(
            lambda symbol: index_by_symbol.at[assignment_map[symbol][0], f"return_{horizon}d_pct"]
            if assignment_map.get(symbol, ("",))[0] in index_by_symbol.index else np.nan
        )
    for source, target in (("current_price", "benchmark_price"), ("price_as_of", "benchmark_price_as_of"),
                           ("price_data_status", "benchmark_price_status")):
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
    result[["benchmark_symbol", "benchmark_name", "benchmark_method"]] = pd.DataFrame(
        assigned.tolist(), index=result.index
    )
    for horizon in (20, 60, 126, 252):
        stock = pd.to_numeric(result.get(f"return_{horizon}d_pct"), errors="coerce")
        benchmark = pd.to_numeric(result.get(f"benchmark_return_{horizon}d_pct"), errors="coerce")
        result[f"excess_return_{horizon}d_pct"] = stock - benchmark
        target = f"benchmark_rs_{horizon}d_percentile"
        result[target] = np.nan
        metric = result[f"excess_return_{horizon}d_pct"]
        for idx in result.index:
            sector = result.get("sector_normalized", result.get("sector", pd.Series("", index=result.index))).fillna("")
            peers = metric[sector.eq(sector.loc[idx])].dropna().index
            if len(peers) < minimum_peers:
                peers = metric[result.benchmark_symbol.eq(result.at[idx, "benchmark_symbol"])].dropna().index
            if len(peers) < minimum_peers:
                peers = metric.dropna().index
            ranks = percentile(metric.loc[peers])
            if idx in ranks:
                result.at[idx, target] = ranks.loc[idx]
    available = result[[f"excess_return_{n}d_pct" for n in (20, 60, 126, 252)]].notna().any(axis=1)
    result["benchmark_rs_status"] = np.where(available, "AVAILABLE_EXPERIMENTAL", "FALLBACK_RAW")
    return result

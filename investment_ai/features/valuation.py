"""Valuation parsing and peer-relative features."""

from __future__ import annotations
import re
from typing import Any
import numpy as np
import pandas as pd
from investment_ai.scoring.common import number, percentile

ALIASES = {
    "trailing_pe": ["Trailing P/E", "TrailingPE", "TrailingPe"],
    "forward_pe": ["Forward P/E", "ForwardPE", "ForwardPe"],
    "peg": ["PEG Ratio (5yr expected)", "PEG Ratio", "PegRatio"],
    "price_sales": ["Price/Sales", "PsRatio"],
    "price_book": ["Price/Book", "PbRatio"],
    "ev_revenue": ["Enterprise Value/Revenue", "EnterpriseValueRevenue"],
    "ev_ebitda": [
        "Enterprise Value/EBITDA",
        "EnterpriseValueEBITDA",
        "EnterprisesValueEBITDARatio",
    ],
    "market_cap": ["Market Cap", "MarketCap"],
}
POSITIVE_ONLY = {"trailing_pe", "forward_pe", "peg", "ev_ebitda", "price_book"}


def normalize_valuation_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def parse_valuation(frame: Any) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    out = {}
    lookup = {normalize_valuation_label(label): label for label in frame.index}
    for key, aliases in ALIASES.items():
        label = next(
            (
                lookup[normalize_valuation_label(alias)]
                for alias in aliases
                if normalize_valuation_label(alias) in lookup
            ),
            None,
        )
        values = (
            pd.to_numeric(frame.loc[label], errors="coerce").dropna()
            if label is not None
            else pd.Series(dtype=float)
        )
        if key in POSITIVE_ONLY:
            values = values[values > 0]
        out[key] = number(values.iloc[0]) if len(values) else np.nan
        history = values.iloc[1:]
        out[f"{key}_history"] = history.tolist()
        out[f"{key}_own_history_percentile"] = (
            float((history >= values.iloc[0]).mean() * 100)
            if len(history) >= 4
            else np.nan
        )
    return out


def add_peer_percentiles(
    frame: pd.DataFrame, metrics: dict[str, bool], minimum_peers: int = 15
) -> pd.DataFrame:
    result = frame.copy()
    result["peer_group_method"] = "universe"
    result["peer_group_size"] = len(result)
    for metric in metrics:
        result[f"{metric}_peer_percentile"] = np.nan
    sectors = result.get(
        "sector_normalized", result.get("sector", pd.Series("", index=result.index))
    ).fillna("")
    for _, indices in result.groupby(sectors).groups.items():
        if len(indices) >= minimum_peers:
            peers, method = indices, "sector"
        else:
            membership = (
                result.loc[indices[0], "index_name"]
                if "index_name" in result and len(indices)
                else ""
            )
            index_peers = result.index[
                result.get("index_name", pd.Series("", index=result.index)).eq(
                    membership
                )
            ]
            peers, method = (
                (index_peers, "index")
                if len(index_peers) >= minimum_peers
                else (result.index, "universe")
            )
        result.loc[indices, "peer_group_method"] = method
        result.loc[indices, "peer_group_size"] = len(peers)
        for metric, higher in metrics.items():
            if metric in result:
                result.loc[indices, f"{metric}_peer_percentile"] = percentile(
                    result.loc[peers, metric], higher
                ).reindex(indices)
    return result

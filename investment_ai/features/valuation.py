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
    for metric in metrics:
        result[f"{metric}_peer_percentile"] = np.nan
        result[f"{metric}_peer_method"] = "universe"
        result[f"{metric}_peer_count"] = 0
    sectors = result.get(
        "sector_normalized", result.get("sector", pd.Series("", index=result.index))
    ).fillna("")
    for _, indices in result.groupby(sectors).groups.items():
        for metric, higher in metrics.items():
            if metric in result:
                sector_valid = pd.to_numeric(result.loc[indices, metric], errors="coerce").dropna()
                membership = ""
                if "index_name" in result and len(indices):
                    membership = sorted(str(result.loc[indices[0], "index_name"]).split("|"))[0].strip()
                memberships = result.get("index_name", pd.Series("", index=result.index)).fillna("").astype(str)
                index_idx = result.index[memberships.map(lambda value: membership in [x.strip() for x in value.split("|")])]
                index_valid = pd.to_numeric(result.loc[index_idx, metric], errors="coerce").dropna()
                universe_valid = pd.to_numeric(result[metric], errors="coerce").dropna()
                if len(sector_valid) >= minimum_peers:
                    peers, method = sector_valid.index, "sector"
                elif len(index_valid) >= minimum_peers:
                    peers, method = index_valid.index, "index"
                else:
                    peers, method = universe_valid.index, "universe"
                result.loc[indices, f"{metric}_peer_method"] = method
                result.loc[indices, f"{metric}_peer_count"] = len(peers)
                result.loc[indices, f"{metric}_peer_percentile"] = percentile(
                    result.loc[peers, metric], higher
                ).reindex(indices)
    result["peer_group_method"] = result.get("forward_pe_peer_method", "universe")
    result["peer_group_size"] = result.get("forward_pe_peer_count", 0)
    return result

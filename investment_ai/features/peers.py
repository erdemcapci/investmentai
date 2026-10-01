"""Metric-specific, security-specific peer group resolution."""

from __future__ import annotations

import pandas as pd


def memberships(value: object) -> tuple[str, ...]:
    """Return stable, de-duplicated memberships from the canonical pipe format."""
    return tuple(sorted({item.strip() for item in str(value).split("|") if item.strip()}))


def resolve_metric_peers(
    frame: pd.DataFrame,
    row_index: object,
    metric: str,
    minimum_peers: int,
) -> tuple[pd.Index, str, str]:
    """Resolve sector, own-index, then universe peers for one security/metric.

    For dual membership, the qualifying index with the most valid observations
    wins.  A count tie is broken by the stable lexical membership name.
    """
    numeric = pd.to_numeric(frame[metric], errors="coerce")
    sector_column = frame.get(
        "sector_normalized", frame.get("sector", pd.Series("", index=frame.index))
    ).fillna("")
    sector = sector_column.loc[row_index]
    sector_peers = numeric[sector_column.eq(sector)].dropna().index
    if len(sector_peers) >= minimum_peers:
        return sector_peers, "sector", ""

    membership_column = frame.get(
        "index_name", pd.Series("", index=frame.index)
    ).fillna("")
    candidates = []
    for membership in memberships(membership_column.loc[row_index]):
        member_mask = membership_column.map(
            lambda value: membership in memberships(value)
        )
        peers = numeric[member_mask].dropna().index
        if len(peers) >= minimum_peers:
            candidates.append((len(peers), membership, peers))
    if candidates:
        count, selected, peers = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        assert count == len(peers)
        return peers, "index", selected

    return numeric.dropna().index, "universe", ""


def fallback_percentile(
    values: pd.Series,
    levels: list[pd.Series],
    minimum_peers: int,
    higher_is_better: bool = True,
) -> pd.Series:
    """Vectorised peer percentile with hierarchical group fallback.

    Each security is ranked inside the first grouping level whose group holds
    at least ``minimum_peers`` valid observations; anything left over is
    ranked against the full universe.  Missing values stay missing.
    """
    from investment_ai.scoring.common import percentile

    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(float("nan"), index=numeric.index, dtype=float)
    assigned = pd.Series(False, index=numeric.index)
    valid = numeric.notna()
    for level in levels:
        keys = level.reindex(numeric.index).fillna("").astype(str)
        counts = valid.groupby(keys).transform("sum")
        eligible = ~assigned & valid & counts.ge(minimum_peers)
        if not eligible.any():
            continue
        ranks = numeric.groupby(keys, group_keys=False).apply(
            lambda group: percentile(group, higher_is_better)
        )
        result.loc[eligible] = ranks.reindex(numeric.index).loc[eligible]
        assigned |= eligible
    remaining = ~assigned & valid
    if remaining.any():
        universe = percentile(numeric, higher_is_better)
        result.loc[remaining] = universe.loc[remaining]
    return result

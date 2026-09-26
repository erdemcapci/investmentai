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

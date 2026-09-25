from __future__ import annotations
from typing import Any
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from investment_ai.scoring.common import curve, weighted, percentile, safe_nanmean


def _ret(s: pd.Series, n: int) -> float:
    return (
        (s.iloc[-1] / s.iloc[-n - 1] - 1) * 100
        if len(s) > n and s.iloc[-n - 1]
        else np.nan
    )


def price_features(history: pd.DataFrame, now: datetime | None = None) -> dict[str, Any]:
    if history is None or history.empty:
        return {}
    h = history.copy()
    h.columns = [str(column).title() for column in h.columns]
    if "Close" not in h:
        return {}
    h.index = pd.to_datetime(h.index, utc=True, errors="coerce")
    h = h.loc[~h.index.isna()].sort_index()
    h = h.loc[~h.index.duplicated(keep="last")]
    now = now or datetime.now(timezone.utc)
    today = pd.Timestamp(now).tz_convert("UTC").date()
    # Daily Yahoo bars stamped today may still be forming; technicals use closes only.
    if len(h) and h.index[-1].date() == today:
        h = h.iloc[:-1]
    close = pd.to_numeric(h["Close"], errors="coerce")
    close = close.where(close > 0).dropna()
    if close.empty:
        return {}
    h = h.reindex(close.index)
    vol = pd.to_numeric(
        h.get("Volume", pd.Series(index=h.index, dtype=float)), errors="coerce"
    ).where(lambda values: values >= 0)
    daily = close.pct_change(fill_method=None)
    age = max((today - close.index[-1].date()).days, 0)
    status = "FRESH" if age <= 4 else "STALE" if age <= 7 else "INSUFFICIENT"
    out = {
        "current_price": close.iloc[-1],
        "price_as_of": close.index[-1].isoformat(),
        "technical_price_as_of": close.index[-1].isoformat(),
        "price_age_calendar_days": age,
        "price_data_status": status,
    }
    for n in (1, 2, 5, 20, 60, 126, 252):
        out[f"return_{n}d_pct"] = _ret(close, n)
    for n in (20, 50, 200):
        ma = close.tail(n).mean() if len(close) >= n else np.nan
        out[f"ma_{n}"] = ma
        out[f"price_vs_ma{n}_pct"] = (close.iloc[-1] / ma - 1) * 100 if ma else np.nan
    for n in (20, 60):
        out[f"drawdown_from_{n}d_high_pct"] = (
            (close.iloc[-1] / close.tail(n).max() - 1) * 100
            if len(close) >= n
            else np.nan
        )
    out["distance_to_52w_high_pct"] = (
        (close.iloc[-1] / close.tail(252).max() - 1) * 100
        if len(close) >= 252
        else np.nan
    )
    out["volatility_20d"] = daily.tail(20).std() * np.sqrt(252) * 100
    out["volatility_60d"] = daily.tail(60).std() * np.sqrt(252) * 100
    out["downside_volatility"] = daily[daily < 0].tail(60).std() * np.sqrt(252) * 100
    wealth = (1 + daily.tail(252).fillna(0)).cumprod()
    out["max_drawdown_1y"] = (
        (wealth / wealth.cummax() - 1).min() * 100 if len(close) >= 252 else np.nan
    )
    out["average_volume_20d"] = vol.tail(20).mean()
    out["average_volume_60d"] = vol.tail(60).mean()
    out["relative_volume_1d"] = (
        vol.iloc[-1] / vol.tail(20).mean() if len(vol) >= 20 else np.nan
    )
    out["relative_volume_5d"] = (
        vol.tail(5).mean() / vol.tail(60).mean() if len(vol) >= 60 else np.nan
    )
    out["average_dollar_volume_20d"] = out["average_volume_20d"] * close.tail(20).mean()
    return out


def add_relative_strength(frame: pd.DataFrame, minimum_peers: int = 15) -> pd.DataFrame:
    result = frame.copy()
    for n in (20, 60, 126, 252):
        result[f"rs_{n}d_percentile"] = np.nan
    sector = result.get("sector_normalized", result.get("sector", pd.Series("", index=result.index))).fillna("")
    for _, idx in result.groupby(sector).groups.items():
        for n in (20, 60, 126, 252):
            metric = f"return_{n}d_pct"
            sector_valid = pd.to_numeric(result.loc[idx, metric], errors="coerce").dropna()
            membership = sorted(str(result.loc[idx[0], "index_name"]).split("|"))[0].strip() if "index_name" in result and len(idx) else ""
            memberships = result.get("index_name", pd.Series("", index=result.index)).fillna("").astype(str)
            index_idx = result.index[memberships.map(lambda value: membership in [x.strip() for x in value.split("|")])]
            index_valid = pd.to_numeric(result.loc[index_idx, metric], errors="coerce").dropna()
            universe_valid = pd.to_numeric(result[metric], errors="coerce").dropna()
            peers = sector_valid.index if len(sector_valid) >= minimum_peers else index_valid.index if len(index_valid) >= minimum_peers else universe_valid.index
            method = "sector" if len(sector_valid) >= minimum_peers else "index" if len(index_valid) >= minimum_peers else "universe"
            result.loc[idx, f"rs_{n}d_peer_method"] = method
            result.loc[idx, f"rs_{n}d_peer_count"] = len(peers)
            result.loc[idx, f"rs_{n}d_percentile"] = percentile(
                result.loc[peers, f"return_{n}d_pct"]
            ).reindex(idx)
    return result


def setup_scores(row: dict[str, Any]) -> dict[str, Any]:
    controlled = curve(
        row.get("drawdown_from_20d_high_pct"),
        [(-30, 0), (-12, 45), (-6, 100), (-2, 75), (0, 35)],
    )
    stability = curve(
        row.get("return_1d_pct"), [(-8, 0), (-2, 35), (0, 80), (2, 100), (6, 45)]
    )
    trend = safe_nanmean(
        [
            curve(
                row.get("price_vs_ma50_pct"), [(-20, 0), (0, 65), (15, 100), (35, 50)]
            ),
            curve(
                row.get("price_vs_ma200_pct"), [(-30, 0), (0, 60), (20, 100), (50, 50)]
            ),
        ]
    )
    pullvol = (
        curve(row.get("relative_volume_5d"), [(0.3, 100), (0.8, 80), (1.2, 45), (2, 0)])
        if row.get("return_5d_pct", 0) < 0
        else 50
    )
    pull, cov, _ = weighted(
        {
            "controlled": controlled,
            "stability": stability,
            "trend": trend,
            "volume": pullvol,
            "rs": row.get("rs_60d_percentile"),
            "recent": stability,
        },
        {
            "controlled": 0.25,
            "stability": 0.20,
            "trend": 0.20,
            "volume": 0.15,
            "rs": 0.10,
            "recent": 0.10,
        },
        0.70,
    )
    breakout = curve(
        row.get("drawdown_from_60d_high_pct"),
        [(-15, 0), (-5, 45), (-1, 90), (1, 100), (8, 55)],
    )
    breakoutvol = curve(
        row.get("relative_volume_1d"), [(0.5, 20), (1, 50), (1.5, 85), (2.5, 100)]
    )
    momentum = safe_nanmean(
        [
            curve(
                row.get("return_20d_pct"),
                [(-10, 0), (0, 40), (10, 90), (25, 100), (50, 30)],
            ),
            curve(
                row.get("return_60d_pct"),
                [(-20, 0), (0, 40), (20, 90), (50, 100), (90, 30)],
            ),
        ]
    )
    alignment = trend
    over = curve(
        row.get("price_vs_ma20_pct"), [(-10, 20), (0, 80), (8, 100), (20, 45), (40, 0)]
    )
    breakout_score, bcov, _ = weighted(
        {
            "breakout": breakout,
            "volume": breakoutvol,
            "momentum": momentum,
            "alignment": safe_nanmean([alignment, over]),
        },
        {"breakout": 0.30, "volume": 0.25, "momentum": 0.25, "alignment": 0.20},
        0.70,
    )
    credible_pull = pd.notna(pull) and pull >= 55 and row.get("return_5d_pct", 0) <= 1
    credible_break = (
        pd.notna(breakout_score)
        and breakout_score >= 55
        and row.get("return_20d_pct", 0) > 0
    )
    if credible_pull and credible_break:
        setup = "MIXED"
        quality = (pull + breakout_score) / 2
    elif credible_pull:
        setup = "PULLBACK"
        quality = pull
    elif credible_break:
        setup = (
            "BREAKOUT"
            if row.get("drawdown_from_20d_high_pct", -9) > -2
            else "MOMENTUM_CONTINUATION"
        )
        quality = breakout_score
    else:
        setup = "NONE"
        quality = (
            np.nan
            if pd.isna(pull) and pd.isna(breakout_score)
            else safe_nanmean([pull, breakout_score])
        )
    status = "INSUFFICIENT_DATA" if pd.isna(quality) else "RANKED"
    return {
        "pullback_setup_score": pull,
        "breakout_momentum_setup_score": breakout_score,
        "setup_coverage": max(cov, bcov),
        "short_term_setup": setup,
        "setup_quality_score": (
            float(np.clip(quality, 0, 100)) if pd.notna(quality) else np.nan
        ),
        "setup_status": status,
    }


def event_timing_score(days_to: Any, days_since: Any = np.nan) -> float:
    if pd.notna(days_to):
        return curve(days_to, [(0, 0), (3, 15), (7, 45), (14, 75), (30, 90), (90, 80)])
    if pd.notna(days_since) and 0 <= days_since <= 3:
        return 40
    return np.nan

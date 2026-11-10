"""Read-only data access for the Streamlit dashboard.

Everything here reads artifacts that ``python main.py`` and ``--backtest``
already wrote; nothing calls Yahoo or writes to the history database.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from investment_ai.runtime import horizon_run_statuses
from investment_ai.scoring.common import NEUTRAL, weighted
from investment_ai.scoring.data_check import stock_data_check
from investment_ai.scoring.overextension import overextension_check

# Pillar columns a viewer can re-weight, with the production weights.
SHORT_TERM_PILLARS = {
    "short_rs_score": ("Relative strength (momentum)", 0.65),
    "expectations_short_score": ("Analyst estimate revisions", 0.20),
    "earnings_drift_score": ("Earnings drift", 0.10),
    "positioning_score": ("Short interest", 0.05),
    "setup_quality_score": ("Setup quality (unweighted label)", 0.0),
    "volume_confirmation_score": ("Signed volume (unweighted label)", 0.0),
    "technical_trend_score": ("Technical trend (unweighted label)", 0.0),
}
LONG_TERM_PILLARS = {
    "quality_score": ("Quality", 0.25),
    "growth_score": ("Growth", 0.20),
    "valuation_score": ("Valuation vs peers", 0.20),
    "expectations_long_score": ("Long-term expectations", 0.20),
    "long_trend_score": ("Long trend", 0.10),
    "financial_safety_score": ("Financial safety", 0.05),
}


def list_runs(runs_dir: Path) -> list[dict]:
    """Completed runs, newest first, with their manifest headline fields."""
    runs = []
    for path in sorted(Path(runs_dir).glob("*"), reverse=True):
        if not (path / "full_analysis.csv").exists():
            continue
        manifest = {}
        if (path / "run_manifest.json").exists():
            try:
                manifest = json.loads((path / "run_manifest.json").read_text())
            except (OSError, json.JSONDecodeError):
                manifest = {}
        runs.append({"run_id": path.name, "path": path, "manifest": manifest})
    return runs


def load_run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full analysis and the exported action list (empty when absent).

    Runs saved before the per-stock data check get it computed on load; it
    depends only on columns the run already stored.
    """
    full = pd.read_csv(path / "full_analysis.csv", low_memory=False)
    action_path = path / "action_list.csv"
    action = pd.read_csv(action_path) if action_path.exists() else pd.DataFrame()
    if "data_check_status" not in full:
        checks = pd.DataFrame(
            [stock_data_check(row) for row in full.to_dict("records")], index=full.index
        )
        full = pd.concat([full, checks], axis=1)
    if "overextended" not in full:
        flags = pd.DataFrame(
            [overextension_check(row) for row in full.to_dict("records")], index=full.index
        )
        full = pd.concat([full, flags], axis=1)
    if len(action) and "overextended" not in action and "symbol" in action:
        lookup = full.drop_duplicates("symbol").set_index("symbol")
        for column in ("data_check_status", "data_check_notes", "overextended", "overextension_note"):
            action[column] = action.symbol.map(lookup[column])
    return full, action


def effective_status(manifest: dict) -> dict:
    """Run status under the current rules.

    Manifests written before coverage notes existed are re-evaluated from the
    coverage metrics they stored, so an old run is not shown as degraded for
    reasons that no longer degrade a run.
    """
    if "coverage_notes" in manifest or "overall_run_status" not in manifest:
        return {
            "status": manifest.get("overall_run_status", "UNKNOWN"),
            "reasons": manifest.get("run_status_reasons", []),
            "notes": manifest.get("coverage_notes", []),
        }
    sources = manifest.get("constituent_sources") or {}
    universe = (
        "INVALID" if any(item.get("source_status") == "INVALID" for item in sources.values())
        else "DEGRADED" if any(item.get("source_status") == "DEGRADED" for item in sources.values())
        else "VALID"
    )
    health = horizon_run_statuses(manifest, universe_valid=universe)
    reasons = (health["common_run_status_reasons"] + health["lt_run_status_reasons"]
               + health["st_run_status_reasons"])
    return {"status": health["overall_run_status"], "reasons": reasons, "notes": health["coverage_notes"]}


def rerank(
    frame: pd.DataFrame, weights: dict[str, float], eligible: pd.Series
) -> pd.DataFrame:
    """Score eligible rows with custom pillar weights (missing pillars count as neutral)."""
    active = {key: weight for key, weight in weights.items() if weight > 0 and key in frame}
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    if active:
        records = frame.loc[eligible, list(active)].to_dict("records")
        values = [weighted(row, active, 0.0, neutral_fill=NEUTRAL)[0] for row in records]
        scores.loc[eligible] = values
    ranks = scores.rank(ascending=False, method="first")
    return pd.DataFrame({"custom_score": scores, "custom_rank": ranks}, index=frame.index)


def apply_filters(
    frame: pd.DataFrame,
    indices: list[str] | None = None,
    sectors: list[str] | None = None,
    min_confidence: float = 0.0,
    exclude_imminent_earnings: bool = False,
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if indices:
        names = frame.get("index_name", pd.Series("", index=frame.index)).fillna("")
        mask &= names.apply(lambda value: any(index in value for index in indices))
    if sectors:
        mask &= frame.get("sector", pd.Series("", index=frame.index)).isin(sectors)
    if min_confidence > 0:
        mask &= pd.to_numeric(frame.get("confidence_score"), errors="coerce").ge(min_confidence)
    if exclude_imminent_earnings and "earnings_imminent_flag" in frame:
        flag = frame["earnings_imminent_flag"].astype(str).str.lower().isin({"true", "1"})
        mask &= ~flag
    return frame[mask]


def best_in_both(frame: pd.DataFrame, top_fraction: float = 0.25) -> pd.DataFrame:
    """Stocks in the top fraction of both the long- and short-term rankings."""
    lt = pd.to_numeric(frame.get("long_term_rank"), errors="coerce")
    st = pd.to_numeric(frame.get("short_term_rank"), errors="coerce")
    if not lt.notna().any() or not st.notna().any():
        return frame.iloc[0:0]
    selected = frame[
        lt.le(lt.max() * top_fraction) & st.le(st.max() * top_fraction)
    ].copy()
    selected["combined_rank_score"] = (
        lt[selected.index] / lt.max() + st[selected.index] / st.max()
    ) / 2
    return selected.sort_values("combined_rank_score")


def price_history(db_path: Path, symbol: str, sessions: int = 400) -> pd.DataFrame:
    """Adjusted daily closes for one symbol from the local price store."""
    if not Path(db_path).exists():
        return pd.DataFrame(columns=["date", "close"])
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        history = pd.read_sql_query(
            "SELECT date, adjusted_close AS close FROM daily_prices "
            "WHERE symbol=? AND adjusted_close IS NOT NULL ORDER BY date DESC LIMIT ?",
            connection,
            params=(symbol, sessions),
        )
    finally:
        connection.close()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)
    for window in (50, 200):
        history[f"ma_{window}"] = history.close.rolling(window).mean()
    return history


def latest_backtest(backtest_dir: Path) -> dict:
    """Newest backtest's summaries and manifest, or an empty dict."""
    candidates = sorted(Path(backtest_dir).glob("backtest-*"), reverse=True)
    for path in candidates:
        if (path / "signal_summary.csv").exists():
            output = {"path": path, "signals": pd.read_csv(path / "signal_summary.csv")}
            strategies = path / "strategy_summary.csv"
            output["strategies"] = pd.read_csv(strategies) if strategies.exists() else pd.DataFrame()
            manifest = path / "backtest_manifest.json"
            output["manifest"] = json.loads(manifest.read_text()) if manifest.exists() else {}
            return output
    return {}


def pillar_breakdown(row: pd.Series, pillars: dict[str, tuple[str, float]]) -> pd.DataFrame:
    """Pillar scores and their weighted contribution versus neutral (50)."""
    rows = []
    for key, (label, weight) in pillars.items():
        value = pd.to_numeric(row.get(key), errors="coerce")
        rows.append(
            {
                "pillar": label,
                "score": value,
                "weight": weight,
                "contribution": weight * (value - NEUTRAL) if pd.notna(value) else np.nan,
            }
        )
    return pd.DataFrame(rows)

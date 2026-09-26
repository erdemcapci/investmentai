"""Point-in-time-safe forward outcome attachment and validation summaries."""

from __future__ import annotations

import sqlite3
from typing import Mapping

import numpy as np
import pandas as pd

HORIZONS = {"5d": 5, "10d": 10, "20d": 20, "3m": 63, "6m": 126, "12m": 252}


def sample_status(n: int) -> str:
    return "INSUFFICIENT_SAMPLE" if n < 30 else "EARLY_SAMPLE" if n < 100 else "USABLE"


def realized_return(history: pd.Series, timestamp: str, sessions: int) -> float:
    """Return from first close on/after prediction time to exactly N later sessions."""
    values = pd.to_numeric(history, errors="coerce").dropna().sort_index()
    values.index = pd.to_datetime(values.index, utc=True)
    eligible = values.loc[values.index >= pd.Timestamp(timestamp)]
    if len(eligible) <= sessions:
        return np.nan
    return float((eligible.iloc[sessions] / eligible.iloc[0] - 1) * 100)


def update_outcomes(connection: sqlite3.Connection, stock_history: Mapping[str, pd.Series],
                    benchmark_history: Mapping[str, pd.Series]) -> int:
    """Attach matured prices to immutable snapshots without ever rebuilding a score."""
    rows = connection.execute(
        "SELECT p.*,o.* FROM prediction_snapshots p JOIN prediction_outcomes o USING(run_id,symbol)"
    ).fetchall()
    names = [item[0] for item in connection.execute(
        "SELECT p.*,o.* FROM prediction_snapshots p JOIN prediction_outcomes o USING(run_id,symbol) LIMIT 0"
    ).description]
    updated = 0
    for raw in rows:
        row = dict(zip(names, raw))
        stock = stock_history.get(row["symbol"])
        benchmark = benchmark_history.get(row.get("index_name"))
        if stock is None:
            continue
        assignments = {}
        for label, sessions in HORIZONS.items():
            column = f"forward_{label}_return"
            if row.get(column) is not None:
                continue
            actual = realized_return(stock, row["run_timestamp"], sessions)
            bench = realized_return(benchmark, row["run_timestamp"], sessions) if benchmark is not None else np.nan
            if pd.notna(actual):
                assignments[column] = actual
                if pd.notna(bench):
                    assignments[f"benchmark_forward_{label}_return"] = bench
                    assignments[f"excess_forward_{label}_return"] = actual - bench
        if assignments:
            with connection:
                connection.execute(
                    "UPDATE prediction_outcomes SET " + ",".join(f"{key}=?" for key in assignments) +
                    " WHERE run_id=? AND symbol=?",
                    [*assignments.values(), row["run_id"], row["symbol"]],
                )
            updated += 1
    return updated


def validation_report(connection: sqlite3.Connection) -> dict:
    frame = pd.read_sql_query(
        "SELECT p.*,o.* FROM prediction_snapshots p JOIN prediction_outcomes o USING(run_id,symbol)", connection
    )
    output: dict[str, dict] = {}
    for horizon in HORIZONS:
        score_column = "short_term_score" if horizon in {"5d", "10d", "20d"} else "long_term_score"
        rank_column = "short_term_rank" if horizon in {"5d", "10d", "20d"} else "long_term_rank"
        return_column = f"forward_{horizon}_return"
        valid = frame.dropna(subset=[score_column, return_column]) if len(frame) else frame
        summary = {"N": len(valid), "status": sample_status(len(valid))}
        if len(valid):
            summary.update({
                "mean": valid[return_column].mean(), "median": valid[return_column].median(),
                "standard_deviation": valid[return_column].std(),
                "win_rate": valid[return_column].gt(0).mean(),
                "rank_correlation": valid[score_column].corr(valid[return_column], method="spearman"),
                "top": {str(n): valid.nsmallest(n, rank_column)[return_column].median() for n in (10, 25, 50)},
                "score_buckets": {
                    str(label): group[return_column].median() for label, group in
                    valid.groupby(pd.cut(valid[score_column], [60, 70, 80, 90, 100], right=False))
                },
            })
        output[horizon] = summary
    return output

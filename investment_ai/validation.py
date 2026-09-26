"""Point-in-time outcome attachment and run-based cross-sectional validation."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from typing import Mapping

import numpy as np
import pandas as pd

from investment_ai.config import TOP_N_MIN_OUTCOME_COVERAGE_PCT

HORIZONS = {"5d": 5, "10d": 10, "20d": 20, "3m": 63, "6m": 126, "12m": 252}
NON_OVERLAP_SPACING = {"5d": 5, "10d": 10, "20d": 20, "3m": 63, "6m": 126, "12m": 252}


def sample_status(n: int) -> str:
    return "INSUFFICIENT_SAMPLE" if n < 20 else "EARLY_SAMPLE" if n < 60 else "USABLE"


def evidence_status(n: int) -> str:
    return {
        "INSUFFICIENT_SAMPLE": "ACCUMULATING",
        "EARLY_SAMPLE": "EARLY_EVIDENCE",
        "USABLE": "USABLE_SAMPLE",
    }[sample_status(n)]


def realized_return(
    history: pd.Series,
    timestamp: str,
    sessions: int,
    baseline_price: float | None = None,
) -> float:
    values = pd.to_numeric(history, errors="coerce").dropna().sort_index()
    values.index = pd.to_datetime(values.index, utc=True)
    baseline = pd.Timestamp(timestamp)
    eligible = values.loc[values.index > baseline]
    if len(eligible) < sessions:
        return np.nan
    if baseline_price is None:
        at_or_after = values.loc[values.index >= baseline]
        if len(at_or_after) <= sessions:
            return np.nan
        baseline_price, future = (
            float(at_or_after.iloc[0]),
            float(at_or_after.iloc[sessions]),
        )
    else:
        future = float(eligible.iloc[sessions - 1])
    return float((future / float(baseline_price) - 1) * 100)


def pending_prediction_symbols(connection: sqlite3.Connection) -> set[str]:
    """Pending history is independent of today's index membership."""
    columns = " OR ".join(f"o.forward_{h}_return IS NULL" for h in HORIZONS)
    return {
        row[0]
        for row in connection.execute(
            f"SELECT DISTINCT p.symbol FROM prediction_snapshots p JOIN prediction_outcomes o "
            f"USING(run_id,symbol) WHERE {columns}"
        )
    }


def update_outcomes(
    connection: sqlite3.Connection,
    stock_history: Mapping[str, pd.Series],
    benchmark_history: Mapping[str, pd.Series],
) -> int:
    where = " OR ".join(f"o.forward_{h}_return IS NULL" for h in HORIZONS)
    query = (
        "SELECT p.*,o.* FROM prediction_snapshots p JOIN prediction_outcomes o "
        f"USING(run_id,symbol) WHERE {where}"
    )
    cursor = connection.execute(query)
    names = [item[0] for item in cursor.description]
    rows, updates, states = cursor.fetchall(), [], []
    now = datetime.now(timezone.utc).isoformat()
    for raw in rows:
        row = dict(zip(names, raw))
        stock = stock_history.get(row["symbol"])
        benchmark = benchmark_history.get(row.get("benchmark_symbol"))
        assignments = {}
        for label, sessions in HORIZONS.items():
            column = f"forward_{label}_return"
            if row.get(column) is not None:
                continue
            baseline_time = pd.Timestamp(row["run_timestamp"])
            if baseline_time.tzinfo is None:
                baseline_time = baseline_time.tz_localize("UTC")
            mature_by_time = pd.Timestamp(now) >= baseline_time + pd.Timedelta(
                days=int(sessions * 7 / 5) + 4
            )
            if not mature_by_time:
                status, reason = "NOT_MATURE", "horizon has not elapsed"
            elif stock is None:
                status, reason = "DELISTED_OR_UNAVAILABLE", "no provider history"
            else:
                actual = realized_return(
                    stock,
                    row.get("price_as_of_utc") or row["run_timestamp"],
                    sessions,
                    row.get("price_at_prediction"),
                )
                if pd.isna(actual):
                    status, reason = "INSUFFICIENT_HISTORY", "insufficient completed sessions"
                else:
                    status, reason, assignments[column] = "AVAILABLE", None, actual
                    bench = (
                        realized_return(
                            benchmark,
                            row.get("benchmark_price_as_of_utc")
                            or row["run_timestamp"],
                            sessions,
                            row.get("benchmark_price_at_prediction"),
                        )
                        if benchmark is not None
                        else np.nan
                    )
                    if pd.notna(bench):
                        assignments[f"benchmark_forward_{label}_return"] = bench
                        assignments[f"excess_forward_{label}_return"] = actual - bench
            states.append((row["run_id"], row["symbol"], label, status, reason, now))
        if assignments:
            updates.append((assignments, row["run_id"], row["symbol"]))
    with connection:
        for assignments, run_id, symbol in updates:
            connection.execute(
                "UPDATE prediction_outcomes SET "
                + ",".join(f"{k}=?" for k in assignments)
                + " WHERE run_id=? AND symbol=?",
                [*assignments.values(), run_id, symbol],
            )
        connection.executemany(
            """INSERT INTO outcome_status VALUES(?,?,?,?,?,?)
            ON CONFLICT(run_id,symbol,horizon) DO UPDATE SET status=excluded.status,
            reason=excluded.reason,updated_at_utc=excluded.updated_at_utc""",
            states,
        )
    return len(updates)


def _per_run(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    score = "short_term_score" if horizon in {"5d", "10d", "20d"} else "long_term_score"
    rank = "short_term_rank" if horizon in {"5d", "10d", "20d"} else "long_term_rank"
    ret, excess = f"forward_{horizon}_return", f"excess_forward_{horizon}_return"
    rows = []
    for run_id, all_rows in frame.groupby("run_id", sort=True):
        ranked = all_rows.dropna(subset=[score, rank])
        if ranked.empty:
            continue
        statuses = ranked.get("outcome_status", pd.Series(index=ranked.index, dtype=object))
        matured = ranked[~statuses.eq("NOT_MATURE")]
        priced = matured[matured[ret].notna() & statuses.loc[matured.index].eq("AVAILABLE")]
        item = {
            "run_id": run_id,
            "analysis_as_of": all_rows.run_timestamp.iloc[0],
            "horizon": horizon,
            "ranked_count": len(ranked),
            "matured_count": len(matured),
            "outcome_coverage_pct": len(priced) / len(matured) * 100 if len(matured) else 0,
        }
        for n in (10, 25, 50):
            # Freeze the selected portfolio before looking at outcomes. Missing
            # members are never replaced by lower-ranked securities.
            selected = ranked.nsmallest(n, rank)
            selected_matured = selected[
                ~selected.get("outcome_status", pd.Series(index=selected.index, dtype=object)).eq("NOT_MATURE")
            ]
            top = selected_matured[selected_matured[ret].notna()]
            selected_count = len(selected_matured)
            coverage = len(top) / selected_count * 100 if selected_count else 0.0
            item[f"top_{n}_selected_count"] = selected_count
            item[f"top_{n}_priced_count"] = len(top)
            item[f"top_{n}_missing_count"] = selected_count - len(top)
            item[f"top_{n}_coverage_pct"] = coverage
            item[f"top_{n}_equal_weight_return"] = (
                top[ret].mean()
                if len(top) and coverage >= TOP_N_MIN_OUTCOME_COVERAGE_PCT
                else np.nan
            )
            item[f"top_{n}_equal_weight_excess_return"] = (
                top[excess].mean()
                if coverage >= TOP_N_MIN_OUTCOME_COVERAGE_PCT
                and excess in top and top[excess].notna().any()
                else np.nan
            )
        item["cross_sectional_spearman_ic"] = (
            priced[score].corr(priced[ret], method="spearman")
            if len(priced) > 1
            and priced[score].nunique() > 1
            and priced[ret].nunique() > 1
            else np.nan
        )
        for low, high in ((60, 70), (70, 80), (80, 90), (90, 100)):
            bucket = priced[
                priced[score].ge(low)
                & priced[score].lt(high if high < 100 else high + 1)
            ]
            item[f"score_bucket_{low}_{high}_return"] = (
                bucket[ret].mean() if len(bucket) else np.nan
            )
        rows.append(item)
    return pd.DataFrame(rows)


def _aggregate(runs: pd.DataFrame, stock_observations: int) -> dict:
    count = len(runs)
    result = {
        "number_of_evaluation_runs": count,
        "evaluation_run_count": count,
        "stock_observation_count": stock_observations,
        "sample_status": sample_status(count),
        "evidence_status": evidence_status(count),
        "overlapping_horizon": True,
    }
    for n in (10, 25, 50):
        values = runs.get(
            f"top_{n}_equal_weight_return", pd.Series(dtype=float)
        ).dropna()
        excess = runs.get(
            f"top_{n}_equal_weight_excess_return", pd.Series(dtype=float)
        ).dropna()
        result.update(
            {
                f"mean_top{n}_return": values.mean() if len(values) else None,
                f"median_top{n}_return": values.median() if len(values) else None,
                f"top{n}_hit_rate": values.gt(0).mean() if len(values) else None,
                f"mean_top{n}_excess": excess.mean() if len(excess) else None,
                f"median_top{n}_excess": excess.median() if len(excess) else None,
                f"top{n}_excess_hit_rate": excess.gt(0).mean() if len(excess) else None,
            }
        )
    ic = runs.get("cross_sectional_spearman_ic", pd.Series(dtype=float)).dropna()
    result.update(
        {
            "mean_ic": ic.mean() if len(ic) else None,
            "median_ic": ic.median() if len(ic) else None,
            "ic_positive_rate": ic.gt(0).mean() if len(ic) else None,
            "ic_standard_deviation": ic.std() if len(ic) else None,
        }
    )
    return result


def validation_report(connection: sqlite3.Connection, mode: str = "ALL_RUNS") -> dict:
    frame = pd.read_sql_query(
        "SELECT p.*,o.* FROM prediction_snapshots p JOIN prediction_outcomes o USING(run_id,symbol)",
        connection,
    )
    frame = frame.loc[:, ~frame.columns.duplicated()]
    output = {}
    for horizon in HORIZONS:
        status_col = (
            "st_run_status" if horizon in {"5d", "10d", "20d"} else "lt_run_status"
        )
        eligible = (
            frame[
                frame.get(status_col, pd.Series(index=frame.index)).isin(
                    ["VALID", "DEGRADED"]
                )
            ]
            if len(frame)
            else frame
        )
        statuses = pd.read_sql_query(
            "SELECT run_id,symbol,status FROM outcome_status WHERE horizon=?",
            connection,
            params=(horizon,),
        )
        if len(eligible):
            eligible = eligible.merge(statuses, on=["run_id", "symbol"], how="left")
            eligible["outcome_status"] = eligible["status"]
            eligible.loc[
                eligible["outcome_status"].isna()
                & eligible[f"forward_{horizon}_return"].notna(),
                "outcome_status",
            ] = "AVAILABLE"
            eligible["outcome_status"] = eligible["outcome_status"].fillna("NOT_MATURE")
        runs = _per_run(eligible, horizon)
        if mode == "NON_OVERLAPPING" and len(runs):
            ordered = runs.sort_values("analysis_as_of")
            selected, last = [], None
            spacing = pd.Timedelta(days=int(NON_OVERLAP_SPACING[horizon] * 7 / 5))
            for index, row in ordered.iterrows():
                current = pd.to_datetime(row["analysis_as_of"], utc=True)
                if last is None or current >= last + spacing:
                    selected.append(index)
                    last = current
            runs = ordered.loc[selected]
        total = len(eligible)
        not_mature = int(eligible.get("outcome_status", pd.Series(dtype=str)).eq("NOT_MATURE").sum())
        matured = total - not_mature
        priced = int(eligible.get("outcome_status", pd.Series(dtype=str)).eq("AVAILABLE").sum())
        unavailable = matured - priced
        summary = _aggregate(runs, priced)
        summary.update(
            {
                "mode": mode,
                "eligible_predictions": total,
                "total_predictions": total,
                "not_mature_predictions": not_mature,
                "matured_predictions": matured,
                "available_outcomes": priced,
                "unavailable_outcomes": unavailable,
                "successfully_priced_outcomes": priced,
                "missing_outcomes": unavailable,
                "unavailable_or_delisted_outcomes": unavailable,
                "coverage_among_matured_pct": priced / matured * 100 if matured else 0,
                "outcome_coverage_pct": priced / matured * 100
                if matured
                else 0,
                "per_run": runs.to_dict("records"),
            }
        )
        # Compatibility: N is now explicitly the primary run sample.
        summary["N"] = summary["evaluation_run_count"]
        summary["status"] = summary["sample_status"]
        output[horizon] = summary
    return output

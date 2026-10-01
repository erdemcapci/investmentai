import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from investment_ai.dashboard_data import (
    apply_filters,
    best_in_both,
    list_runs,
    pillar_breakdown,
    price_history,
    rerank,
)


def _frame():
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "index_name": ["S&P 500", "STOXX Europe 600", "S&P 500|STOXX Europe 600", "S&P 500"],
            "sector": ["Tech", "Energy", "Tech", "Energy"],
            "confidence_score": [90, 50, 80, 70],
            "earnings_imminent_flag": [False, True, "False", "True"],
            "long_term_rank": [1, 2, 3, 4],
            "short_term_rank": [2, np.nan, 1, 4],
            "short_rs_score": [90, 40, 60, np.nan],
            "earnings_drift_score": [50, np.nan, 90, 70],
        }
    )


def test_list_runs_keeps_completed_runs_newest_first(tmp_path):
    for name, complete in (("20260101T000000Z-a", True), ("20260102T000000Z-b", True), ("20260103T000000Z-c", False)):
        directory = tmp_path / name
        directory.mkdir()
        if complete:
            (directory / "full_analysis.csv").write_text("symbol\nA\n")
            (directory / "run_manifest.json").write_text(json.dumps({"overall_run_status": "VALID"}))
    runs = list_runs(tmp_path)
    assert [run["run_id"] for run in runs] == ["20260102T000000Z-b", "20260101T000000Z-a"]
    assert runs[0]["manifest"]["overall_run_status"] == "VALID"


def test_filters_by_index_sector_confidence_and_earnings():
    frame = _frame()
    assert apply_filters(frame, ["STOXX Europe 600"]).symbol.tolist() == ["B", "C"]
    assert apply_filters(frame, sectors=["Energy"]).symbol.tolist() == ["B", "D"]
    assert apply_filters(frame, min_confidence=75).symbol.tolist() == ["A", "C"]
    assert apply_filters(frame, exclude_imminent_earnings=True).symbol.tolist() == ["A", "C"]


def test_rerank_uses_custom_weights_with_neutral_fill():
    frame = _frame()
    eligible = frame.short_term_rank.notna()
    momentum_only = rerank(frame, {"short_rs_score": 1.0}, eligible)
    assert momentum_only.custom_rank.tolist()[:1] == [1.0]
    assert np.isnan(momentum_only.custom_score.iloc[1])  # B is not eligible
    # With no data for any weighted pillar a stock gets no score at all...
    assert np.isnan(momentum_only.custom_score.iloc[3])
    # ...but a missing pillar next to a present one counts as neutral (50).
    blended = rerank(frame, {"short_rs_score": 0.5, "earnings_drift_score": 0.5}, eligible)
    assert blended.custom_score.iloc[3] == pytest.approx((50 + 70) / 2)
    drift_only = rerank(frame, {"short_rs_score": 0.0, "earnings_drift_score": 1.0}, eligible)
    assert drift_only.sort_values("custom_rank").index[0] == 2  # C has the strongest drift


def test_best_in_both_requires_both_horizons():
    frame = pd.DataFrame(
        {"symbol": list("ABCDEFGH"), "long_term_rank": range(1, 9), "short_term_rank": [8, 1, 2, 3, 4, 5, 6, 7]}
    )
    assert best_in_both(frame).symbol.tolist() == ["B"]


def test_pillar_breakdown_contribution_versus_neutral():
    row = pd.Series({"quality_score": 80, "growth_score": np.nan})
    data = pillar_breakdown(row, {"quality_score": ("Quality", 0.25), "growth_score": ("Growth", 0.2)})
    assert data.contribution.iloc[0] == pytest.approx(7.5)
    assert np.isnan(data.contribution.iloc[1])


def test_price_history_reads_read_only_with_moving_averages(tmp_path):
    db = tmp_path / "history.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE daily_prices(symbol TEXT, date TEXT, adjusted_close REAL)")
    connection.executemany(
        "INSERT INTO daily_prices VALUES (?,?,?)",
        [("A", str(day.date()), float(i)) for i, day in enumerate(pd.bdate_range("2025-01-01", periods=60), 1)],
    )
    connection.commit()
    connection.close()
    history = price_history(db, "A")
    assert len(history) == 60 and history.close.iloc[-1] == 60
    assert history.ma_50.iloc[-1] == pytest.approx(np.mean(range(11, 61)))
    assert history.ma_200.isna().all()
    assert price_history(tmp_path / "missing.sqlite", "A").empty

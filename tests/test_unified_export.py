import numpy as np
import pandas as pd

from investment_ai.reporting.export import build_investment_ranking, export_run


def _analysis() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": "both",
                "symbol": "BOTH",
                "company_name": "Both Horizons",
                "index_name": "S&P 500",
                "sector": "Technology",
                "long_term_rank": 4,
                "long_term_rank_change_7d": 2,
                "long_term_score": 81,
                "long_term_score_change_7d": 1.5,
                "short_term_rank": 7,
                "short_term_rank_change_7d": -1,
                "short_term_score": 73,
                "short_term_score_change_7d": -0.5,
                "short_term_setup": "PULLBACK",
                "target_mean": 120,
                "target_mean_change_30d_pct": np.nan,
                "rank_history_status": "AVAILABLE",
            },
            {
                "security_id": "lt",
                "symbol": "LT",
                "company_name": "Long Only",
                "index_name": "S&P 500",
                "sector": "Industrials",
                "long_term_rank": 2,
                "long_term_score": 78,
                "short_term_rank": np.nan,
                "short_term_score": np.nan,
                "rank_history_status": "HISTORY_NOT_YET_AVAILABLE",
            },
            {
                "security_id": "st",
                "symbol": "ST",
                "company_name": "Short Only",
                "index_name": "STOXX Europe 600",
                "sector": "Financials",
                "long_term_rank": np.nan,
                "long_term_score": np.nan,
                "short_term_rank": 1,
                "short_term_score": 76,
                "short_term_setup": "BREAKOUT",
                "rank_history_status": "HISTORY_NOT_YET_AVAILABLE",
            },
            {
                "security_id": "none",
                "symbol": "NONE",
                "company_name": "Unranked",
                "index_name": "S&P 500",
                "sector": "Utilities",
                "long_term_rank": np.nan,
                "long_term_score": np.nan,
                "short_term_rank": np.nan,
                "short_term_score": np.nan,
            },
        ]
    )


def test_unified_ranking_includes_both_lt_only_and_st_only_once():
    output = build_investment_ranking(_analysis(), "VALID")
    assert output.symbol.tolist() == ["BOTH", "ST", "LT"]
    assert output.symbol.is_unique
    both = output.set_index("symbol").loc["BOTH"]
    assert both.long_term_rank == 4
    assert both.short_term_rank == 7
    assert output.set_index("symbol").loc["LT", "long_term_rank"] == 2
    assert output.set_index("symbol").loc["ST", "short_term_rank"] == 1


def test_unified_ranking_preserves_history_and_never_fills_missing_with_zero():
    output = build_investment_ranking(_analysis(), "DEGRADED").set_index("symbol")
    assert output.loc["BOTH", "long_term_rank_change_7d"] == 2
    assert output.loc["BOTH", "long_term_score_change_7d"] == 1.5
    assert output.loc["BOTH", "short_term_rank_change_7d"] == -1
    assert output.loc["BOTH", "short_term_score_change_7d"] == -0.5
    assert pd.isna(output.loc["BOTH", "target_mean_change_30d_pct"])
    assert output.loc["LT", "rank_history_status"] == "HISTORY_NOT_YET_AVAILABLE"
    assert set(output.run_status) == {"DEGRADED"}


def test_unified_ranking_merges_duplicate_security_memberships():
    duplicate = pd.DataFrame(
        [
            {
                "security_id": "same", "symbol": "ABC", "company_name": "ABC",
                "index_name": "S&P 500", "long_term_rank": 3,
                "long_term_score": 80, "short_term_rank": np.nan,
                "short_term_score": np.nan,
            },
            {
                "security_id": "same", "symbol": "ABC", "company_name": "ABC",
                "index_name": "STOXX Europe 600", "long_term_rank": np.nan,
                "long_term_score": np.nan, "short_term_rank": 5,
                "short_term_score": 70,
            },
        ]
    )
    output = build_investment_ranking(duplicate, "VALID")
    assert len(output) == 1
    assert output.loc[0, "long_term_rank"] == 3
    assert output.loc[0, "short_term_rank"] == 5
    assert output.loc[0, "index_name"] == "S&P 500 | STOXX Europe 600"


def test_unified_ranking_has_no_combined_score():
    columns = set(build_investment_ranking(_analysis(), "VALID").columns)
    assert "combined_score" not in columns
    assert "overall_score" not in columns
    assert "overall_investment_score" not in columns


def test_export_preserves_horizon_files_and_health_gates_unified_file(tmp_path):
    full = _analysis()
    lt = full[full.long_term_rank.notna()]
    st = full[full.short_term_rank.notna()]
    methodology = pd.DataFrame([{"score": "test", "formula": "unchanged"}])
    export_run(
        tmp_path, full, lt, st,
        {"overall_run_status": "VALID"}, methodology,
    )
    assert (tmp_path / "long_term_ranking.csv").exists()
    assert (tmp_path / "short_term_ranking.csv").exists()
    assert (tmp_path / "investment_ranking.csv").exists()
    assert not (tmp_path / "investment_ranking_invalid_diagnostic.csv").exists()

    export_run(
        tmp_path, full, lt, st,
        {"overall_run_status": "INVALID"}, methodology,
        lt_status="INVALID", st_status="INVALID",
    )
    assert not (tmp_path / "investment_ranking.csv").exists()
    assert (tmp_path / "investment_ranking_invalid_diagnostic.csv").exists()

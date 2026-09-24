import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import main
import scanner_data
from scoring import RANKED


def combined_source_frame():
    return pd.DataFrame([
        {"symbol": "AAPL", "security": "Apple", "gics_sector": "Technology", "index_name": "S&P 500"},
        {"symbol": "VRTX", "security": "Vertex", "gics_sector": "Health Care", "index_name": "S&P 500"},
        {"symbol": "SAP.DE", "security": "SAP", "gics_sector": "Technology", "index_name": "STOXX Europe 600"},
    ])


def ranking_rows():
    symbols = ["AAPL", "DIS", "UBER", "VRTX", "SAP.DE", "ASML.AS", "SIE.DE", "AIR.PA"]
    scores = [72, 91, 82, 76, 95, 88, 84, 79]
    rows = []
    for symbol, score in zip(symbols, scores):
        rows.append({
            "symbol": symbol,
            "company_name": symbol,
            "index_name": "STOXX Europe 600" if "." in symbol else "S&P 500",
            "analyst_long_term_status": RANKED,
            "short_term_status": RANKED,
            "analyst_long_term_score": float(score),
            "short_term_entry_score": float(100 - abs(84 - score)),
            "positive_rating_pct": 80.0,
            "rating_count": 20,
            "analyst_long_term_data_coverage_pct": 100.0,
            "short_term_data_coverage_pct": 100.0,
            "overall_data_quality": "HIGH",
            "candidate_profile_priority": 1,
            "combined_score": score * .6 + (100 - abs(84 - score)) * .4,
            "candidate_profile": "STRONG CANDIDATE",
        })
    return pd.DataFrame(rows)


def analysis_inputs():
    constituents = pd.DataFrame([{
        "symbol": "AAPL", "company_name": "Apple", "sector": "Technology",
        "index_name": "S&P 500",
    }])
    prices = pd.DataFrame([{
        "symbol": "AAPL", "history_price": 100, "ma_20": 99, "ma_50": 98,
        "ma_200": 90, "price_vs_200d_ma_pct": 11.1, "volatility_annual_pct": 25,
        "average_volume_20d": 1_000_000, "average_dollar_volume_20d": 100_000_000,
        "history_error": "",
    }])
    analysts = pd.DataFrame([{
        "symbol": "AAPL", "latest_quote": 101, "target_current": 101,
        "target_low": 110, "target_mean": 130, "target_median": 125,
        "target_high": 145, "strong_buy": 10, "buy": 8, "hold": 2,
        "sell": 0, "strong_sell": 0, "eps_up_7d": 1, "eps_up_30d": 2,
        "eps_down_7d": 0, "eps_down_30d": 0,
        "next_earnings_date": "2026-12-01T00:00:00Z", "data_errors": "",
    }])
    return constituents, prices, analysts


def test_canonical_loader_uses_both_indexes_and_preserves_vertex():
    with patch.object(main, "fetch_index_constituents", return_value=combined_source_frame()) as fetch:
        result = main.download_combined_constituents()
    fetch.assert_called_once_with(main.CONSTITUENT_CACHE_DIR)
    assert result.symbol.tolist() == ["AAPL", "VRTX", "SAP.DE"]
    assert result.index_name.tolist() == ["S&P 500", "S&P 500", "STOXX Europe 600"]
    assert result.company_name.tolist() == ["Apple", "Vertex", "SAP"]


def test_canonical_loader_requires_both_indexes():
    only_sp = combined_source_frame().query("index_name == 'S&P 500'")
    with patch.object(main, "fetch_index_constituents", return_value=only_sp):
        try:
            main.download_combined_constituents()
        except RuntimeError as exc:
            assert "Both constituent sources are required" in str(exc)
        else:
            raise AssertionError("S&P-only execution was accepted")


def test_duplicate_symbols_cannot_be_downloaded_twice():
    duplicate = pd.concat([combined_source_frame(), combined_source_frame().iloc[[0]]], ignore_index=True)
    duplicate.loc[3, "index_name"] = "STOXX Europe 600"
    with patch.object(main, "fetch_index_constituents", return_value=duplicate):
        universe = main.download_combined_constituents()
    symbols = universe.symbol.tolist()
    assert len(symbols) == len(set(symbols))
    assert symbols.count("AAPL") == 1
    assert universe.set_index("symbol").loc["AAPL", "index_name"] == (
        "S&P 500 | STOXX Europe 600"
    )


def test_deduplication_keeps_first_available_metadata():
    sparse = pd.DataFrame([
        {"symbol": "DUAL", "security": pd.NA, "index_name": "S&P 500"},
        {"symbol": "DUAL", "security": "Dual Company", "index_name": "STOXX Europe 600"},
    ])
    result = scanner_data.combine_index_constituents([sparse])
    assert result.loc[0, "company_name"] == "Dual Company"
    assert result.loc[0, "index_name"] == "S&P 500 | STOXX Europe 600"


def test_index_name_survives_analysis_merge():
    result = main.build_analysis(*analysis_inputs())
    assert result.loc[0, "index_name"] == "S&P 500"


def test_global_rankings_mix_regions_and_keep_membership():
    long_term, short_term, combined, *_ = main.create_rankings(ranking_rows())
    for result in (long_term, short_term, combined):
        assert "index_name" in result
        assert result.index_name.str.contains("S&P 500", regex=False).any()
        assert result.index_name.str.contains("STOXX Europe 600", regex=False).any()
    assert long_term.symbol.iloc[0] == "SAP.DE"
    assert list(long_term.symbol[:4]) == ["SAP.DE", "DIS", "ASML.AS", "SIE.DE"]
    assert set(short_term.symbol[:3]) == {"UBER", "SIE.DE", "ASML.AS"}
    assert combined.combined_rank.tolist() == list(range(1, len(combined) + 1))


def test_rank_is_global_not_grouped_by_index():
    frame = ranking_rows()
    long_term, short_term, combined, *_ = main.create_rankings(frame)
    assert long_term.symbol.tolist().index("SAP.DE") < long_term.symbol.tolist().index("DIS")
    assert long_term.symbol.tolist().index("DIS") < long_term.symbol.tolist().index("AAPL")
    assert len(short_term) == len(frame) == len(combined)


def test_no_cli_index_selector_top_100_or_vertex_detail_hook():
    source = Path(main.__file__).read_text()
    assert "--indexes" not in source
    assert "TICKER_TO_CHECK" not in source
    assert "print_ticker_details" not in source
    assert "top_100" not in source.lower()
    assert list(inspect.signature(main.main).parameters) == []


def test_fixed_analysis_values_are_unchanged_by_membership_column():
    constituents, prices, analysts = analysis_inputs()
    sp = main.build_analysis(constituents, prices, analysts)
    eu = constituents.assign(index_name="STOXX Europe 600")
    eu_result = main.build_analysis(eu, prices, analysts)
    score_inputs = ["selected_target_upside_pct", "rating_count", "positive_rating_pct", "target_dispersion_pct"]
    pd.testing.assert_series_equal(sp.loc[0, score_inputs], eu_result.loc[0, score_inputs])
    assert np.isclose(sp.loc[0, "selected_target_upside_pct"], (125 / 101 - 1) * 100)


def test_original_dual_score_regression_for_fixed_mocked_input():
    analysis = main.build_analysis(*analysis_inputs())
    short = pd.DataFrame([{
        "symbol": "AAPL", "return_1d_pct": 1, "return_2d_pct": .5,
        "return_5d_pct": -3, "return_20d_pct": 4,
        "distance_from_ma20_pct": 2, "distance_from_ma50_pct": 3,
        "short_volatility_20d_pct": 22,
        "volatility_20d_annualized_pct": 22,
        "drawdown_from_20d_high_pct": -5, "negative_days_last_5": 2,
        "worst_daily_return_5d_pct": -2,
    }])
    scored = main.calculate_dual_scores(analysis, short).iloc[0]
    assert np.isclose(scored.analyst_long_term_score, 81.090904, atol=1e-6)
    assert np.isclose(scored.short_term_entry_score, 89.195833, atol=1e-6)
    assert np.isclose(scored.combined_score, 84.332876, atol=1e-6)
    assert scored.index_name == "S&P 500"

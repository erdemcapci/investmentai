"""v3.0.1 provider-contract, integrity, and regression coverage."""

from datetime import datetime, timedelta, timezone
import math
import numpy as np
import pandas as pd
import pytest
import main as app
from investment_ai.data.cache import JsonCache
from investment_ai.data.history_store import HistoryStore
from investment_ai.features.analyst import (
    parse_recommendations,
    parse_targets,
)
from investment_ai.features.fundamentals import (
    derive_fundamentals,
    normalize_statement_label,
)
from investment_ai.features.technical import price_features, setup_scores
from investment_ai.features.valuation import normalize_valuation_label, parse_valuation
from investment_ai.pipeline import normalize_sector, _history_scores
from investment_ai.scoring.long_term import score_long_term
from investment_ai.scoring.short_term import score_short_term


@pytest.mark.parametrize(
    "values",
    [
        ["S&P 500", "STOXX Europe 600"],
        ["S&P 500 | STOXX Europe 600"],
        ["S&P 500", "S&P 500 | STOXX Europe 600", "STOXX Europe 600"],
    ],
)
def test_combined_universe_membership_shapes(monkeypatch, values):
    monkeypatch.setattr(
        app,
        "fetch_index_constituents",
        lambda *_: pd.DataFrame({"symbol": range(len(values)), "index_name": values}),
    )
    assert len(app.download_combined_constituents()) == len(values)


def test_combined_universe_missing_source_raises(monkeypatch):
    monkeypatch.setattr(
        app,
        "fetch_index_constituents",
        lambda *_: pd.DataFrame({"symbol": ["A"], "index_name": ["S&P 500"]}),
    )
    with pytest.raises(RuntimeError):
        app.download_combined_constituents()


def realistic_recommendations():
    return pd.DataFrame(
        {
            "period": ["0m", "-1m", "-2m", "-3m"],
            "strongBuy": [8, 7, 6, 5],
            "buy": [4, 4, 4, 4],
            "hold": [2, 3, 4, 5],
            "sell": [1, 1, 1, 1],
            "strongSell": [0, 0, 0, 0],
        }
    )


def test_recommendations_real_period_column_contract():
    result = parse_recommendations(realistic_recommendations())
    assert result["ratings_valid_current"] is True
    assert result["rating_count_current"] == 15
    assert result["positive_rating_change_3m_pp"] > 0


@pytest.mark.parametrize("column", ["strongBuy", "buy", "hold", "sell", "strongSell"])
def test_each_missing_recommendation_count_remains_unknown(column):
    frame = realistic_recommendations()
    frame.loc[0, column] = np.nan
    result = parse_recommendations(frame)
    assert result["ratings_valid_current"] is False
    assert math.isnan(result["rating_count_current"])
    assert math.isnan(result["recommendation_strength_current"])


@pytest.mark.parametrize("bad", [-1, -0.1, "bad", np.inf])
def test_invalid_recommendation_counts_rejected(bad):
    frame = realistic_recommendations().astype(object)
    frame.loc[0, "sell"] = bad
    assert not parse_recommendations(frame)["ratings_valid_current"]


@pytest.mark.parametrize(
    "label",
    [
        "Total Revenue",
        "TotalRevenue",
        "total_revenue",
        "total-revenue",
        "total/revenue",
    ],
)
def test_statement_label_normalization(label):
    assert normalize_statement_label(label) == "totalrevenue"


def raw_statements(capex=-10):
    dates = [
        pd.Timestamp("2023-12-31"),
        pd.Timestamp("2024-12-31"),
        pd.Timestamp("2022-12-31"),
        pd.Timestamp("2021-12-31"),
    ]
    income = pd.DataFrame(
        {
            dates[0]: [100, 15, 10, 20, 18, 22, 2, 3, 12],
            dates[1]: [120, 24, 15, 30, 26, 32, 2, 4, 16],
            dates[2]: [80, 8, 5, 12, 10, 14, 2, 2, 8],
            dates[3]: [60, 5, 3, 9, 7, 10, 2, 1, 5],
        },
        index=[
            "TotalRevenue",
            "OperatingIncome",
            "NetIncomeCommonStockholders",
            "GrossProfit",
            "EBIT",
            "NormalizedEBITDA",
            "InterestExpenseNonOperating",
            "TaxProvision",
            "PretaxIncome",
        ],
    )
    balance = pd.DataFrame(
        {dates[1]: [20, 30, 60, 200], dates[0]: [18, 28, 55, 180]},
        index=[
            "CashCashEquivalentsAndShortTermInvestments",
            "TotalDebt",
            "TotalEquityGrossMinorityInterest",
            "TotalAssets",
        ],
    )
    cash = pd.DataFrame(
        {dates[1]: [30, capex], dates[0]: [20, -8]},
        index=["TotalCashFromOperatingActivities", "CapitalExpenditures"],
    )
    return income, balance, cash


def test_camelcase_statements_and_reversed_dates_use_latest():
    result = derive_fundamentals(*raw_statements())
    assert result["revenue"] == 120
    assert result["operating_income"] == 24
    assert result["free_cash_flow"] == 20
    assert result["revenue_growth_yoy"] == 20
    assert result["revenue_cagr_3y"] > 20


@pytest.mark.parametrize("capex,expected", [(-10, 20), (10, 20)])
def test_capex_both_sign_conventions(capex, expected):
    assert derive_fundamentals(*raw_statements(capex))["free_cash_flow"] == expected


def test_zero_revenue_prevents_fcf_margin():
    income, balance, cash = raw_statements()
    income.loc["TotalRevenue", pd.Timestamp("2024-12-31")] = 0
    assert math.isnan(derive_fundamentals(income, balance, cash)["fcf_margin_pct"])


@pytest.mark.parametrize("label", ["Forward P/E", "ForwardPE", "ForwardPe"])
def test_forward_pe_aliases(label):
    assert (
        parse_valuation(pd.DataFrame({"Current": [18]}, index=[label]))["forward_pe"]
        == 18
    )


@pytest.mark.parametrize("label", ["PEG Ratio (5yr expected)", "PEG Ratio", "PegRatio"])
def test_peg_aliases(label):
    assert (
        parse_valuation(pd.DataFrame({"Current": [1.5]}, index=[label]))["peg"] == 1.5
    )


@pytest.mark.parametrize("label", ["Forward P/E", "forward_pe", "Forward-PE"])
def test_valuation_label_normalization(label):
    assert normalize_valuation_label(label) == "forwardpe"


@pytest.mark.parametrize(
    "label", ["Forward P/E", "PEG Ratio", "Enterprise Value/EBITDA", "Price/Book"]
)
def test_nonpositive_valuation_ratios_invalid(label):
    result = parse_valuation(pd.DataFrame({"Current": [-1]}, index=[label]))
    key = {
        "Forward P/E": "forward_pe",
        "PEG Ratio": "peg",
        "Enterprise Value/EBITDA": "ev_ebitda",
        "Price/Book": "price_book",
    }[label]
    assert math.isnan(result[key])


def test_own_history_lower_is_better_direction_and_minimum():
    frame = pd.DataFrame(
        {"Current": [8], "Q1": [10], "Q2": [11], "Q3": [12], "Q4": [13]},
        index=["Forward P/E"],
    )
    assert parse_valuation(frame)["forward_pe_own_history_percentile"] == 100
    assert math.isnan(
        parse_valuation(frame.iloc[:, :4])["forward_pe_own_history_percentile"]
    )


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Technology", "Technology"),
        ("Information Technology", "Technology"),
        ("Tech", "Technology"),
        ("Financial Services", "Financials"),
        ("Banks", "Financials"),
        ("Insurance", "Financials"),
        ("Consumer Cyclical", "Consumer Discretionary"),
    ],
)
def test_sector_normalization(source, expected):
    assert normalize_sector(source) == expected


def test_missing_setup_is_unknown_not_low():
    result = setup_scores({})
    assert math.isnan(result["setup_quality_score"])
    assert result["setup_status"] == "INSUFFICIENT_DATA"


def test_long_term_quality_core_gate():
    row = {
        "expectations_score": 80,
        "forward_pe_peer_percentile": 80,
        "forward_revenue_growth": 0.2,
        "forward_eps_growth": 0.2,
        "rs_126d_percentile": 80,
        "rs_252d_percentile": 80,
        "price_vs_ma200_pct": 5,
    }
    assert math.isnan(score_long_term(row)["long_term_score"])


def test_short_term_core_gate_requires_setup():
    row = {
        "expectations_score": 80,
        "expectations_coverage": 1,
        "rs_20d_percentile": 80,
        "rs_60d_percentile": 80,
        "rs_126d_percentile": 80,
    }
    assert math.isnan(score_short_term(row)["short_term_score"])


def test_financial_quality_can_reach_full_coverage_without_fcf():
    row = {
        "is_financial": True,
        "return_on_equity": 30,
        "return_on_assets": 5,
        "net_margin_pct": 25,
        "earnings_growth_yoy": 30,
        "revenue_growth_yoy": 20,
        "forward_revenue_growth": 0.2,
        "forward_eps_growth": 0.2,
        "revenue_cagr_3y": 15,
        "growth_acceleration": 5,
        "forward_pe_peer_percentile": 80,
        "price_book_peer_percentile": 80,
        "peg_peer_percentile": 80,
        "forward_pe_own_history_percentile": 80,
        "price_book_own_history_percentile": 80,
        "expectations_score": 80,
        "rs_126d_percentile": 80,
        "rs_252d_percentile": 80,
        "price_vs_ma200_pct": 5,
        "market_risk_score": 20,
    }
    result = score_long_term(row)
    assert result["quality_coverage"] == 1
    assert pd.notna(result["quality_score"])


def test_target_history_score_prefers_median():
    score, _ = _history_scores(
        {"target_median_change_7d_pct": 20, "target_mean_change_7d_pct": -50}
    )
    assert score > 50


def test_no_target_history_no_momentum():
    score, revenue = _history_scores({})
    assert math.isnan(score) and math.isnan(revenue)


def test_cache_validator_preserves_old_component(tmp_path):
    cache = JsonCache(tmp_path)
    old, _ = cache.get_or_fetch(
        "analyst_targets",
        "A",
        0,
        lambda: {"target_mean": 100},
        validator=lambda x: bool(x),
    )
    stale, status = cache.get_or_fetch(
        "analyst_targets",
        "A",
        0,
        lambda: {"target_mean": np.nan},
        validator=lambda x: pd.notna(x["target_mean"]),
    )
    assert stale == old and status.startswith("stale_cache")


def test_same_observation_timestamp_not_duplicated(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    when = datetime.now(timezone.utc)
    assert store.upsert_analyst("A", {"target_mean": 10}, when)
    assert not store.upsert_analyst("A", {"target_mean": 11}, when)
    assert store.db.execute("select count(*) from analyst_snapshots").fetchone()[0] == 1


def test_target_and_revenue_history_only_after_elapsed_days(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    now = datetime.now(timezone.utc)
    store.upsert_analyst(
        "A", {"target_median": 100, "revenue_0y_avg": 1000}, now - timedelta(days=31)
    )
    row = store.add_analyst_history_features(
        "A", {"target_median": 110, "revenue_0y_avg": 1200}, now
    )
    assert row["target_median_change_30d_pct"] == 10
    assert row["revenue_0y_change_30d_pct"] == 20
    assert math.isnan(row["target_median_change_90d_pct"])


def test_price_duplicate_timezone_and_invalid_close_handling():
    index = [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    ]
    result = price_features(
        pd.DataFrame({"close": [10, 11, -1], "volume": [1, 2, 3]}, index=index)
    )
    assert result["current_price"] == 11
    assert "+00:00" in result["price_as_of"]


def test_partial_price_history_suppresses_long_metrics():
    result = price_features(
        pd.DataFrame(
            {"Close": np.arange(1, 101), "Volume": 10},
            index=pd.date_range("2025-01-01", periods=100),
        )
    )
    assert math.isnan(result["return_252d_pct"])
    assert math.isnan(result["ma_200"])
    assert math.isnan(result["distance_to_52w_high_pct"])


def test_invalid_target_range_and_zero_price():
    target = parse_targets({"low": 120, "mean": 110, "median": 110, "high": 100})
    assert target["target_low"] > target["target_high"]


def test_canonical_main_mocked_integration(monkeypatch, tmp_path):
    universe = pd.DataFrame(
        {
            "symbol": ["AAPL", "SAP.DE"],
            "index_name": ["S&P 500", "STOXX Europe 600"],
            "sector": ["Technology", "Technology"],
        }
    )
    monkeypatch.setattr(app, "download_combined_constituents", lambda: universe)
    monkeypatch.setattr(app, "download_prices", lambda *_: pd.DataFrame())
    monkeypatch.setattr(
        app,
        "build_price_features",
        lambda *_: pd.DataFrame(
            {"symbol": universe.symbol, "current_price": [100, 100]}
        ),
    )
    monkeypatch.setattr(
        app.YahooClient,
        "fetch_many",
        lambda *_: pd.DataFrame(
            {"symbol": universe.symbol, "analyst_component_error_count": 0}
        ),
    )
    full = universe.assign(
        long_term_score=[80, 70],
        short_term_score=[60, 65],
        long_term_rank=[1, 2],
        short_term_rank=[2, 1],
        quality_score=80,
        growth_score=70,
        valuation_score=60,
        expectations_score=75,
    )
    monkeypatch.setattr(app, "build_analysis", lambda *_: (full, full, full))
    monkeypatch.setattr(app, "HISTORY_DB", tmp_path / "history.db")
    monkeypatch.setattr(app, "print_rankings", lambda lt, st, top: None)
    monkeypatch.setattr(app, "EXPORT_RESULTS", False)
    app.main()

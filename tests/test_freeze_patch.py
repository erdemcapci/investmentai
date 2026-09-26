"""Production-freeze correctness regressions."""

from datetime import datetime, timezone
import sqlite3

import numpy as np
import pandas as pd

import main
from investment_ai.data.history_store import HistoryStore
from investment_ai.data.symbol_resolver import SymbolResolver
from investment_ai.features.technical import price_features
from investment_ai.runtime import horizon_run_statuses
from investment_ai.validation import _aggregate


def _mapping_row(**updates):
    row = {
        "security_id": "ISIN:DE1", "listing_id": "listing", "source_index": "STOXX Europe 600",
        "source_symbol": "SAP", "company_name": "SAP SE", "country": "Germany",
        "exchange": None, "isin": "DE1", "canonical_yahoo_symbol": "SAP.DE",
        "mapping_method": "EXCHANGE_SUFFIX_HEURISTIC", "mapping_status": "HEURISTIC",
        "mapping_confidence": .65, "mapping_verified_at_utc": None, "mapping_error": None,
    }
    row.update(updates)
    return row


def test_constituent_resolution_never_calls_metadata(monkeypatch, tmp_path):
    frame = pd.DataFrame([
        {"symbol": "AAPL", "source_symbol": "AAPL", "security": "Apple", "country": "United States", "index_name": "S&P 500"},
        {"symbol": "SAP", "source_symbol": "SAP", "security": "SAP SE", "country": "Germany", "index_name": "STOXX Europe 600"},
    ])
    monkeypatch.setattr(main, "fetch_index_constituents", lambda *_: frame)
    monkeypatch.setattr(main, "HISTORY_DB", tmp_path / "history.db")
    monkeypatch.setattr(main.YahooClient, "metadata_lookup", lambda *_: (_ for _ in ()).throw(AssertionError("live lookup")))
    result = main.download_combined_constituents()
    assert result.loc[result.source_symbol.eq("SAP"), "mapping_status"].item() == "HEURISTIC"


def test_provider_info_upgrades_or_rejects_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "HISTORY_DB", tmp_path / "history.db")
    universe = pd.DataFrame([{**_mapping_row(), "symbol": "SAP.DE"}])
    good = pd.DataFrame([{"symbol": "SAP.DE", "info_success": True,
                          "provider_info_longName": "SAP SE", "provider_info_country": "Germany",
                          "provider_info_quoteType": "EQUITY"}])
    assert main._verify_provider_mappings(universe, good).mapping_status.item() == "VERIFIED"
    universe = pd.DataFrame([{**_mapping_row(source_symbol="BAD", security_id="ISIN:BAD"), "symbol": "SAP.DE"}])
    bad = good.assign(provider_info_longName="Unrelated Holdings")
    assert main._verify_provider_mappings(universe, bad).mapping_status.item() == "AMBIGUOUS"


def test_unavailable_verification_degrades_but_does_not_invalidate(monkeypatch):
    monkeypatch.setattr(main, "SP500_MIN_CONSTITUENTS", 1, raising=False)
    universe = pd.DataFrame([{
        **_mapping_row(), "index_name": "STOXX Europe 600",
        "constituent_list_fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }])
    # Test policy directly with the production floor represented by enough rows.
    universe = pd.concat([universe] * 560, ignore_index=True)
    health = main._constituent_health(universe, datetime.now(timezone.utc))["stoxx600"]
    assert health["source_status"] == "DEGRADED"


def test_persisted_verified_mapping_needs_no_lookup(tmp_path):
    db = sqlite3.connect(tmp_path / "m.db")
    resolver = SymbolResolver(db)
    first = resolver.resolve("STOXX Europe 600", "SAP", "SAP SE", "Germany", isin="DE1")
    resolver.verify(first.as_dict(), {"longName": "SAP SE", "country": "Germany", "quoteType": "EQUITY"})
    resolver.metadata_lookup = lambda *_: (_ for _ in ()).throw(AssertionError())
    assert resolver.resolve("STOXX Europe 600", "SAP", "SAP SE", "Germany", isin="DE1").mapping_status == "VERIFIED"


def test_price_freshness_metrics_drive_common_and_st_health():
    metrics = {"price_present_coverage_pct": 100, "price_coverage_pct": 100,
               "price_fresh_coverage_pct": 50}
    health = horizon_run_statuses(metrics)
    assert health["common_run_status"] == health["st_run_status"] == "INVALID"


def test_friday_close_is_fresh_on_saturday_and_old_close_is_not():
    friday = pd.DataFrame({"Close": np.arange(1, 31), "Volume": 10},
                          index=pd.bdate_range(end="2026-09-25", periods=30, tz="UTC"))
    assert price_features(friday, datetime(2026, 9, 26, tzinfo=timezone.utc))["price_data_status"] == "FRESH"
    old = friday.set_axis(pd.bdate_range(end="2026-09-14", periods=30, tz="UTC"))
    assert price_features(old, datetime(2026, 9, 26, tzinfo=timezone.utc))["price_data_status"] == "INSUFFICIENT"


def test_validation_counts_only_evaluable_runs():
    immature = pd.DataFrame([{"matured_count": 0, "top_10_selected_count": 10, "top_10_matured_count": 0,
                               **{f"top_{n}_equal_weight_return": np.nan for n in (10, 25, 50)},
                               "cross_sectional_spearman_ic": np.nan} for _ in range(60)])
    result = _aggregate(immature, 0)
    assert (result["historical_run_count"], result["matured_run_count"], result["evaluation_run_count"]) == (60, 0, 0)
    assert result["sample_status"] == "INSUFFICIENT_SAMPLE"
    mature = immature.iloc[:20].copy()
    mature["matured_count"] = 10
    mature["top_10_matured_count"] = 10
    mature.loc[:7, "top_10_equal_weight_return"] = 1.0
    assert _aggregate(mature, 80)["top10_evaluation_run_count"] == 8
    usable = pd.concat([mature.assign(top_10_equal_weight_return=1.0)] * 3, ignore_index=True)
    assert _aggregate(usable, 600)["sample_status"] == "USABLE"


def test_prediction_v5_metadata_round_trip_and_minimal_row(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    metadata = {"symbol": "SAP.DE", "security_id": "ISIN:DE1", "benchmark_symbol": "^STOXX",
                "benchmark_name": "STOXX 600", "benchmark_return_basis": "ADJUSTED_CLOSE_RETURN",
                "benchmark_currency": "EUR", "benchmark_assignment_method": "REGIONAL_INDEX",
                "trading_currency": "EUR", "financial_statement_currency": "EUR", "market_cap_currency": "EUR"}
    store.save_predictions("r1", "2026-01-01T00:00:00+00:00", "3.1.2", [metadata])
    store.save_predictions("r2", "2026-01-02T00:00:00+00:00", "3.1.2", [{"symbol": "OLD"}])
    row = dict(store.db.execute("SELECT * FROM prediction_snapshots WHERE run_id='r1'").fetchone())
    for key, value in metadata.items():
        assert row[key] == value
    assert store.db.execute("SELECT security_id FROM prediction_snapshots WHERE run_id='r2'").fetchone()[0] is None


def test_resume_retries_only_failed_provider_row(monkeypatch, tmp_path):
    context = main.RunContext.create("run", tmp_path)
    pd.DataFrame({"symbol": ["A", "B", "C"], "provider_success": [1, 0, 1]}).to_csv(context.directory / "normalized_provider.csv", index=False)
    requested = []
    def fetch(_self, rows):
        requested.extend(rows.symbol.tolist())
        return pd.DataFrame({"symbol": rows.symbol, "provider_success": 1})
    monkeypatch.setattr(main.YahooClient, "fetch_many", fetch)
    result = main._provider_data(context, pd.DataFrame({"symbol": ["A", "B", "C"]}), main.configure_logging(context.directory, "run"))
    assert requested == ["B"]
    assert result.set_index("symbol").at["B", "provider_success"] == 1
    assert context.checkpoint["provider_symbols_failed"] == []


def test_two_normal_runs_use_real_orchestration_and_incremental_storage(monkeypatch, tmp_path):
    """Run execute twice; only network boundaries and tiny-universe floor are replaced."""
    symbols = ["AAA", "BBB", "CCC.DE", "DDD.DE"]
    universe = pd.DataFrame({
        "symbol": symbols, "security_id": symbols,
        "index_name": ["S&P 500", "S&P 500", "STOXX Europe 600", "STOXX Europe 600"],
        "sector": ["Technology"] * 4, "mapping_status": ["VERIFIED"] * 4,
    })
    monkeypatch.setattr(main, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(main, "HISTORY_DB", tmp_path / "history.db")
    monkeypatch.setattr(main, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(main, "download_combined_constituents", lambda: universe.copy())
    monkeypatch.setattr(main, "print_rankings", lambda *_: None)
    monkeypatch.setattr(main, "_constituent_health", lambda *_: {
        "sp500": {"source_status": "VALID", "status_reasons": [], "mapping_success_pct": 100},
        "stoxx600": {"source_status": "VALID", "status_reasons": [], "mapping_success_pct": 100},
    })
    price_windows = []
    def prices(boundary_symbols, start, end):
        price_windows.append((start, end, tuple(boundary_symbols)))
        days = pd.bdate_range(start=start, end=end, tz="UTC")
        return pd.concat({symbol: pd.DataFrame({"Close": np.linspace(80, 120, len(days)),
                                                "Volume": np.full(len(days), 1_000_000)}, index=days)
                          for symbol in boundary_symbols}, axis=1)
    monkeypatch.setattr(main, "download_prices_range", prices)

    endpoint_calls = {name: 0 for name in ("info", "analyst", "valuation", "fundamental")}
    cached_provider = None
    def provider(_self, rows):
        nonlocal cached_provider
        if cached_provider is not None:
            return cached_provider[cached_provider.symbol.isin(rows.symbol)].copy()
        for key in endpoint_calls:
            endpoint_calls[key] += len(rows)
        base = {
            "provider_success": 1, "info_success": True,
            "sector_raw_yahoo": "Technology", "target_low": 90, "target_mean": 130,
            "target_median": 125, "target_high": 150, "positive_rating_pct": 80,
            "rating_count": 20, "forward_eps_growth": .2, "forward_revenue_growth": .15,
            "fcf_margin_pct": 18, "operating_margin_pct": 22,
            "operating_margin_change_1y": 2, "fcf_growth_yoy": 12,
            "cash_conversion": 1.1, "return_on_equity": 20, "roic": 18,
            "revenue_growth_yoy": 12, "revenue_cagr_3y": 10, "growth_acceleration": 2,
            "forward_pe": 20, "ev_ebitda": 12, "peg": 1.5, "price_book": 4,
            "forward_pe_own_history_percentile": 60, "free_cash_flow": 100,
            "market_cap": 1000, "financial_statement_currency": "USD",
            "market_cap_currency": "USD", "debt_to_equity": .5,
            "net_debt_to_ebitda": 1, "interest_coverage": 8,
            "days_to_next_earnings": 30, "days_since_last_earnings": 60,
            "positive_rating_change_1m_pp": 2, "positive_rating_change_3m_pp": 3,
            "rating_action_balance_7d": .2, "rating_action_balance_30d": .2,
            "positive_surprise_rate_4q": 75, "eps_surprise_volatility_4q": 5,
            "data_errors": "",
        }
        for period in ("0q", "plus_1q", "0y", "plus_1y"):
            for days in (7, 30, 60, 90):
                base[f"eps_{period}_change_{days}d_pct"] = 2
            for days in (7, 30):
                base[f"eps_revision_breadth_{period}_{days}d"] = .3
        for days in (7, 30, 90):
            base[f"target_median_change_{days}d_pct"] = 2
        for period in ("0q", "plus_1q", "0y", "plus_1y"):
            for days in (7, 30, 90):
                base[f"revenue_{period}_change_{days}d_pct"] = 2
        for component in (*main.YahooClient.ANALYST_COMPONENTS, "earnings_dates", "info", "valuation", "fundamental"):
            base[f"{component}_cache_status"] = "fresh_cache"
        cached_provider = pd.DataFrame([{"symbol": symbol, **base} for symbol in symbols])
        return cached_provider.copy()
    monkeypatch.setattr(main.YahooClient, "fetch_many", provider)

    original_analysis = main.build_analysis
    score_runs = []
    def observed_analysis(*args):
        result = original_analysis(*args)
        score_runs.append(result[0][["symbol", "long_term_score", "short_term_score"]])
        return result
    monkeypatch.setattr(main, "build_analysis", observed_analysis)

    assert main.execute("offline-1") == 0
    assert main.execute("offline-2") == 0
    assert len(score_runs) == 2 and all(frame.long_term_score.notna().all() for frame in score_runs)
    assert len(price_windows) >= 2
    assert price_windows[-1][0] > price_windows[0][0]
    assert endpoint_calls == {"info": 4, "analyst": 4, "valuation": 4, "fundamental": 4}
    db = sqlite3.connect(tmp_path / "history.db")
    assert db.execute("SELECT COUNT(DISTINCT run_id) FROM prediction_snapshots").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM analyst_observations").fetchone()[0] == 0
    for run_id in ("offline-1", "offline-2"):
        directory = tmp_path / "runs" / run_id
        assert (directory / "run_manifest.json").exists()
        assert (directory / "long_term_ranking.csv").exists()
        assert (directory / "short_term_ranking.csv").exists()

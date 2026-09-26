from datetime import date
import sqlite3

import numpy as np
import pandas as pd
import pytest

import main
from investment_ai.data.constituents import fetch_stoxx600_constituents
from investment_ai.data.price_store import PriceStore
from investment_ai.data.symbol_resolver import SymbolResolver
from investment_ai.data.yahoo import currency_metadata
from investment_ai.pipeline import apply_fcf_currency_guard
from investment_ai.pipeline import build_analysis
from investment_ai.validation import _per_run, validation_report
from investment_ai.data.history_store import HistoryStore


@pytest.mark.parametrize(
    "raw,country,canonical",
    [
        ("ATCOa", "Sweden", "ATCO-A.ST"),
        ("ERICb", "Sweden", "ERIC-B.ST"),
        ("HMB", "Sweden", "HM-B.ST"),
        ("SRENH", "Switzerland", "SREN.SW"),
        ("VOW3", "Germany", "VOW3.DE"),
        ("AIR", "France", "AIR.PA"),
        ("SHEL", "United Kingdom", "SHEL.L"),
    ],
)
def test_raw_stoxx_resolution_end_to_end(raw, country, canonical):
    resolver = SymbolResolver(sqlite3.connect(":memory:"))
    assert resolver.resolve("STOXX Europe 600", raw, raw, country).canonical_yahoo_symbol == canonical


def test_provider_identity_mismatch_and_verified_persistence():
    db = sqlite3.connect(":memory:")
    mismatch = SymbolResolver(db, lambda _: {"longName": "Different Corp", "country": "France", "quoteType": "EQUITY"})
    assert mismatch.resolve("STOXX Europe 600", "AIR", "Airbus SE", "France").mapping_status == "AMBIGUOUS"

    calls = []
    verified = SymbolResolver(sqlite3.connect(":memory:"), lambda symbol: calls.append(symbol) or {
        "longName": "Airbus SE", "country": "France", "quoteType": "EQUITY"
    })
    assert verified.resolve("STOXX Europe 600", "AIR", "Airbus SE", "France").mapping_status == "VERIFIED"
    assert verified.resolve("STOXX Europe 600", "AIR", "Airbus SE", "France").mapping_method == "PERSISTED_VERIFIED"
    assert calls == ["AIR.PA"]


def test_raw_stoxx_does_not_drop_unsupported_country(monkeypatch, tmp_path):
    class Response:
        text = "unused"
        def raise_for_status(self):
            return None

    monkeypatch.setattr("investment_ai.data.constituents.requests.get", lambda *a, **k: Response())
    monkeypatch.setattr("investment_ai.data.constituents.pd.read_html", lambda *_a, **_k: [pd.DataFrame({
        "Company": ["Known", "Unsupported"], "Ticker": ["AIR", "XYZ"],
        "Country": ["France", "Atlantis"],
    })])
    result = fetch_stoxx600_constituents(tmp_path / "stoxx.csv")
    assert result.source_symbol.tolist() == ["AIR", "XYZ"]


def test_currency_metadata_and_fcf_guard():
    assert currency_metadata({"currency": "USD", "financialCurrency": "USD"}) == {
        "trading_currency": "USD", "market_cap_currency": "USD",
        "financial_statement_currency": "USD",
    }
    frame = pd.DataFrame({
        "market_cap": [100, 100, 100], "free_cash_flow": [10, 10, 10],
        "market_cap_currency": ["USD", "USD", None],
        "financial_statement_currency": ["USD", "EUR", "USD"],
    })
    result = apply_fcf_currency_guard(frame)
    assert result.fcf_yield.iloc[0] == pytest.approx(0.1)
    assert np.isnan(result.fcf_yield.iloc[1]) and np.isnan(result.fcf_yield.iloc[2])
    assert result.fcf_yield_currency_status.tolist() == [
        "COMPATIBLE", "CURRENCY_MISMATCH", "CURRENCY_UNAVAILABLE"
    ]
    assert result.fcf_yield.notna().any()  # regression: valid metadata prevents global loss


def test_yahoo_sector_is_canonical_for_financial_applicability():
    universe = pd.DataFrame({
        "symbol": ["BANK"], "index_name": ["STOXX Europe 600"],
        "sector": ["Industrials"],
    })
    prices = pd.DataFrame({"symbol": ["BANK"], "current_price": [10]})
    provider = pd.DataFrame({
        "symbol": ["BANK"], "sector_raw_yahoo": ["Financial Services"],
        "market_cap": [100], "free_cash_flow": [10], "roic": [12],
        "net_debt_to_ebitda": [2],
    })
    full, _, _ = build_analysis(universe, prices, provider)
    row = full.iloc[0]
    assert row.sector == "Financials"
    assert bool(row.is_financial) and not bool(row.roic_applicable)
    assert np.isnan(row.roic) and np.isnan(row.net_debt_to_ebitda)


def test_top_n_selection_is_frozen_before_outcome_filtering():
    frame = pd.DataFrame({
        "run_id": ["r"] * 11, "run_timestamp": ["2025-01-01T00:00:00Z"] * 11,
        "short_term_score": list(range(99, 88, -1)), "short_term_rank": list(range(1, 12)),
        "forward_5d_return": [1, np.nan] + [1] * 9,
        "excess_forward_5d_return": [0, np.nan] + [0] * 9,
        "outcome_status": ["AVAILABLE", "PRICE_MISSING"] + ["AVAILABLE"] * 9,
    })
    run = _per_run(frame, "5d").iloc[0]
    assert run.top_10_selected_count == 10
    assert run.top_10_priced_count == 9
    assert run.top_10_missing_count == 1
    assert run.top_10_equal_weight_return == pytest.approx(1)


def test_low_coverage_cohort_is_excluded_but_diagnostic_remains():
    frame = pd.DataFrame({
        "run_id": ["r"] * 10, "run_timestamp": ["2025-01-01T00:00:00Z"] * 10,
        "short_term_score": range(10), "short_term_rank": range(1, 11),
        "forward_5d_return": [1] * 4 + [np.nan] * 6,
        "outcome_status": ["AVAILABLE"] * 4 + ["PRICE_MISSING"] * 6,
    })
    run = _per_run(frame, "5d").iloc[0]
    assert run.top_10_coverage_pct == 40
    assert np.isnan(run.top_10_equal_weight_return)


def test_maturity_coverage_uses_only_matured_predictions(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    rows = [{"symbol": f"S{i}", "short_term_score": 50, "short_term_rank": i + 1} for i in range(100)]
    store.save_predictions("r", "2025-01-01T00:00:00+00:00", "3.1.1", rows)
    with store.db:
        for i in range(55):
            store.db.execute("UPDATE prediction_outcomes SET forward_5d_return=1 WHERE run_id='r' AND symbol=?", (f"S{i}",))
        states = []
        for i in range(100):
            status = "AVAILABLE" if i < 55 else "PRICE_MISSING" if i < 60 else "NOT_MATURE"
            states.append(("r", f"S{i}", "5d", status, None, "2026-01-01"))
        store.db.executemany("INSERT INTO outcome_status VALUES(?,?,?,?,?,?)", states)
    summary = validation_report(store.db)["5d"]
    assert (summary["total_predictions"], summary["not_mature_predictions"], summary["matured_predictions"]) == (100, 40, 60)
    assert (summary["available_outcomes"], summary["unavailable_outcomes"]) == (55, 5)
    assert summary["coverage_among_matured_pct"] == pytest.approx(91.6667)


def test_incremental_price_history_uses_store_and_combines(monkeypatch):
    db = sqlite3.connect(":memory:")
    store = PriceStore(db)
    calls = []
    def fake(symbols, start, end):
        calls.append((symbols, start, end))
        days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
        return pd.concat({s: pd.DataFrame({"Close": range(10, 10 + len(days)), "Volume": range(1, 1 + len(days))}, index=days) for s in symbols}, axis=1)
    monkeypatch.setattr(main, "download_prices_range", fake)
    first, _ = main._incremental_price_history(store, {"A": "AAA"}, date(2025, 1, 1), date(2025, 1, 10))
    assert calls[0][1] == date(2025, 1, 1)
    calls.clear()
    second, _ = main._incremental_price_history(store, {"A": "AAA"}, date(2025, 1, 1), date(2025, 1, 10))
    assert calls[0][1] == date(2025, 1, 5)  # five-calendar-day revision overlap
    assert len(first["AAA"].dropna(how="all")) == 10
    assert len(second["AAA"].dropna(how="all")) >= 2
    assert store.history("A").raw_close.isna().all()

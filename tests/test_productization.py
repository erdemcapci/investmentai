from datetime import datetime, timezone
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from investment_ai.data.cache import JsonCache
from investment_ai.data.history_store import HistoryStore
from investment_ai.data.yahoo import aggregate_component_status
from investment_ai.features.benchmark import add_benchmark_relative_strength, assign_benchmark
from investment_ai.features.technical import setup_scores
from investment_ai.runtime import RunContext, build_manifest, run_status
from investment_ai.pipeline import target_metrics
from investment_ai.scoring.common import percentile
from investment_ai.status import ERROR, FRESH_CACHE, FRESH_PROVIDER, PARTIAL, STALE_FALLBACK
from investment_ai.validation import realized_return, sample_status
import main as app


def test_cache_v2_is_rejected_and_refreshed(tmp_path):
    cache = JsonCache(tmp_path)
    cache._path("estimate", "ABC").write_text(json.dumps({
        "cache_schema_version": 2, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {"forward_eps_growth": 20},
    }))
    item, status = cache.get_or_fetch("estimate", "ABC", 24,
                                      lambda: {"forward_eps_growth": .2})
    assert status == FRESH_PROVIDER
    assert item["cache_schema_version"] == 3
    assert item["data"]["forward_eps_growth"] == .2


@pytest.mark.parametrize("statuses,expected", [
    ([STALE_FALLBACK, ERROR, FRESH_CACHE], PARTIAL),
    ([STALE_FALLBACK, FRESH_CACHE], STALE_FALLBACK),
    ([FRESH_CACHE, FRESH_CACHE], FRESH_CACHE),
    ([FRESH_CACHE, FRESH_PROVIDER], FRESH_PROVIDER),
])
def test_analyst_status_precedence(statuses, expected):
    assert aggregate_component_status(statuses) == expected


def test_setup_none_is_not_ranked():
    result = setup_scores({
        "drawdown_from_20d_high_pct": -30, "drawdown_from_60d_high_pct": -15,
        "return_1d_pct": -8, "return_5d_pct": -5, "return_20d_pct": -10,
        "return_60d_pct": -20, "price_vs_ma20_pct": -10,
        "price_vs_ma50_pct": -20, "price_vs_ma200_pct": -30,
        "relative_volume_1d": .5, "relative_volume_5d": 2, "rs_60d_percentile": 0,
    })
    assert result["short_term_setup"] == "NONE"
    assert result["setup_status"] == "NO_CREDIBLE_SETUP"
    assert setup_scores({})["setup_status"] == "INSUFFICIENT_DATA"


def test_percentile_endpoints_single_and_ties():
    values = pd.Series(range(15))
    assert percentile(values).iloc[[0, -1]].tolist() == [0, 100]
    assert percentile(values, False).iloc[[0, -1]].tolist() == [100, 0]
    assert percentile(pd.Series([9])).iloc[0] == 50
    tied = percentile(pd.Series([1, 2, 2, 3]))
    assert tied.iloc[1] == tied.iloc[2] == 50


def test_target_upside_is_independent_of_range_and_mean_fallback():
    frame = pd.DataFrame({
        "target_median": [190, 190, np.nan], "target_mean": [180, 180, 180],
        "target_low": [np.nan, 210, np.nan], "target_high": [np.nan, 200, np.nan],
        "current_price": [150, 150, 150],
    })
    result = target_metrics(frame)
    assert result.target_upside_pct.tolist() == pytest.approx([26.6666667, 26.6666667, 20])
    assert result.target_dispersion_pct.isna().all()


@pytest.mark.parametrize("metrics,expected", [
    ({"price_coverage_pct": 98, "quality_coverage_pct": 80, "valuation_coverage_pct": 80, "expectations_coverage_pct": 80}, "VALID"),
    ({"price_coverage_pct": 95, "quality_coverage_pct": 60, "valuation_coverage_pct": 60, "expectations_coverage_pct": 65}, "DEGRADED"),
    ({"price_coverage_pct": 89, "quality_coverage_pct": 80, "valuation_coverage_pct": 80, "expectations_coverage_pct": 80}, "INVALID"),
])
def test_run_quality_gates(metrics, expected):
    assert run_status(metrics)[0] == expected


def test_manifest_has_versions_coverage_and_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = RunContext.create("run-1", tmp_path, False)
    manifest = build_manifest(context, {"price_coverage_pct": 99, "universe_count": 2}, {"top_n": 10})
    assert manifest["application_version"] == "1.0.2"
    assert manifest["scoring_model_version"] == "3.1.1"
    assert manifest["cache_schema_version"] == 3
    assert manifest["database_schema_version"] == 4
    assert manifest["price_coverage_pct"] == 99
    assert json.loads((context.directory / "run_manifest.json").read_text())["config"]["top_n"] == 10


def test_database_migration_and_future_rejection(tmp_path):
    path = tmp_path / "history.db"
    store = HistoryStore(path)
    assert store.db.execute("SELECT value FROM metadata WHERE key='database_schema_version'").fetchone()[0] == "4"
    store.close()
    db = sqlite3.connect(path)
    db.execute("UPDATE metadata SET value='999' WHERE key='database_schema_version'")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="future database schema"):
        HistoryStore(path)


def test_benchmark_assignment_excess_return_and_regions():
    assert assign_benchmark("S&P 500")[0] == "^GSPC"
    assert assign_benchmark("STOXX Europe 600")[0] == "^STOXX"
    assert assign_benchmark("STOXX Europe 600|S&P 500")[0] == "^GSPC"
    frame = pd.DataFrame({
        "index_name": ["S&P 500", "STOXX Europe 600"], "sector": ["Tech", "Tech"],
        "return_20d_pct": [12, 7], "benchmark_return_20d_pct": [10, 3],
    })
    result = add_benchmark_relative_strength(frame, minimum_peers=2)
    assert result.excess_return_20d_pct.tolist() == [2, 4]
    assert result.benchmark_symbol.tolist() == ["^GSPC", "^STOXX"]


def test_forward_return_uses_trading_sessions_and_maturity_guards():
    dates = pd.bdate_range("2026-01-01", periods=8, tz="UTC")
    history = pd.Series(range(100, 108), index=dates)
    expected = (105 / 100 - 1) * 100
    assert realized_return(history, dates[0].isoformat(), 5) == pytest.approx(expected)
    assert np.isnan(realized_return(history, dates[0].isoformat(), 10))
    assert [sample_status(n) for n in (29, 30, 100)] == [
        "INSUFFICIENT_SAMPLE", "EARLY_SAMPLE", "USABLE"
    ]


def test_resume_skips_success_and_retries_failure(tmp_path, monkeypatch):
    context = RunContext.create("resume-me", tmp_path)
    context.checkpoint["provider_symbols_complete"] = ["DONE"]
    context.checkpoint["provider_symbols_failed"] = ["RETRY"]
    context.save_checkpoint()
    calls = []

    def fetch(_client, pending):
        calls.append(pending.symbol.tolist())
        return pd.DataFrame({"symbol": pending.symbol, "provider_success": [1] * len(pending)})

    monkeypatch.setattr(app.YahooClient, "fetch_many", fetch)
    logger = app.configure_logging(context.directory, context.run_id)
    result = app._provider_data(
        context, pd.DataFrame({"symbol": ["DONE", "RETRY"], "sector": ["", ""]}), logger
    )
    assert calls == [["DONE", "RETRY"]]
    assert set(result.symbol) == {"DONE", "RETRY"}
    assert context.checkpoint["provider_symbols_complete"] == ["DONE", "RETRY"]


def test_replay_uses_stored_inputs_without_yahoo(tmp_path, monkeypatch):
    source = tmp_path / "captured"
    source.mkdir()
    universe = pd.DataFrame({"symbol": ["ABC"], "index_name": ["S&P 500"], "sector": ["Tech"]})
    prices = pd.DataFrame({"symbol": ["ABC"], "current_price": [100]})
    provider = pd.DataFrame({"symbol": ["ABC"], "analyst_cache_status": [FRESH_CACHE]})
    universe.to_csv(source / "universe.csv", index=False)
    prices.to_csv(source / "price_features.csv", index=False)
    provider.to_csv(source / "normalized_provider.csv", index=False)
    full = universe.assign(
        long_term_score=80, short_term_score=70, long_term_rank=1, short_term_rank=1,
        quality_score=80, growth_score=80, valuation_score=80,
        expectations_long_score=80, risk_score=20, confidence_score=90,
        short_term_setup="PULLBACK", benchmark_rs_status="FALLBACK_RAW",
    )
    full.to_csv(source / "full_analysis.csv", index=False)
    (source / "run_manifest.json").write_text(json.dumps({
        "application_version": "1.0.2", "scoring_model_version": "3.1.1",
        "scoring_code_fingerprint": app.scoring_code_fingerprint(),
        "artifact_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("universe.csv", "price_features.csv", "normalized_provider.csv", "full_analysis.csv")
        },
    }))
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "HISTORY_DB", tmp_path / "history.sqlite")
    monkeypatch.setattr(app, "build_analysis", lambda *_: (full, full, full))
    monkeypatch.setattr(app, "print_rankings", lambda *_: None)
    monkeypatch.setattr(app.YahooClient, "fetch_many", lambda *_: pytest.fail("Yahoo called in replay"))
    assert app.execute("captured", replay=True) == 0
    replayed = pd.read_csv(next(tmp_path.glob("captured-replay-*/full_analysis.csv")))
    assert replayed.long_term_score.iloc[0] == 80

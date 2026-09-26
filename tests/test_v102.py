from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

import main as app
from investment_ai.data.history_store import HistoryStore
from investment_ai.runtime import (
    RunContext,
    atomic_json,
    build_manifest,
    scoring_code_fingerprint,
)
from investment_ai.validation import validation_report


def test_atomic_json_recursively_sanitizes_scientific_values(tmp_path):
    path = tmp_path / "safe.json"
    atomic_json(
        path,
        {
            "a": np.nan,
            "b": np.inf,
            "c": -np.inf,
            "d": np.float64(1.5),
            "e": np.int64(3),
            "f": pd.NA,
            "g": pd.NaT,
            "nested": {"x": [1, np.nan, pd.Timestamp("2026-01-02T03:04:05Z")]},
        },
    )
    value = json.loads(path.read_text())
    assert value == {
        "a": None,
        "b": None,
        "c": None,
        "d": 1.5,
        "e": 3,
        "f": None,
        "g": None,
        "nested": {"x": [1, None, "2026-01-02T03:04:05+00:00"]},
    }
    assert not any(token in path.read_text() for token in ("NaN", "Infinity"))


def test_manifest_and_validation_json_accept_nan_metrics(tmp_path):
    context = RunContext.create("nan-run", tmp_path)
    manifest = build_manifest(
        context,
        {
            "risk_median": np.nan,
            "confidence_median": np.nan,
            "validation": {"rank_correlation": np.nan},
        },
        {},
    )
    assert manifest["risk_median"] is np.nan or np.isnan(manifest["risk_median"])
    saved = json.loads((context.directory / "run_manifest.json").read_text())
    assert saved["risk_median"] is saved["confidence_median"] is None
    assert saved["validation"]["rank_correlation"] is None


def _frame():
    return pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "long_term_score": 80.0,
                "long_term_rank": 1,
                "short_term_score": 70.0,
                "short_term_rank": 2,
                "short_term_setup": "PULLBACK",
            }
        ]
    )


@pytest.mark.parametrize(
    "lt,st,expected",
    [
        ("INVALID", "VALID", (None, 70.0)),
        ("VALID", "INVALID", (80.0, None)),
        ("INVALID", "INVALID", None),
        ("DEGRADED", "DEGRADED", (80.0, 70.0)),
    ],
)
def test_health_gated_history(lt, st, expected, tmp_path):
    rows = app._persistence_rows(_frame(), {"lt_run_status": lt, "st_run_status": st})
    store = HistoryStore(tmp_path / "history.db")
    if rows:
        store.save_rankings("run", "2026-01-01T00:00:00+00:00", rows)
        store.save_predictions(
            "run", "2026-01-01T00:00:00+00:00", "3.1.2", rows, lt, st, "DEGRADED"
        )
    count = store.db.execute("SELECT count(*) FROM prediction_snapshots").fetchone()[0]
    if expected is None:
        assert count == 0
        assert (
            store.db.execute("SELECT count(*) FROM ranking_history").fetchone()[0] == 0
        )
    else:
        row = store.db.execute(
            "SELECT long_term_score,short_term_score,lt_run_status,st_run_status "
            "FROM prediction_snapshots"
        ).fetchone()
        assert tuple(row[:2]) == expected
        assert tuple(row[2:]) == (lt, st)
    store.close()


def test_validation_filters_invalid_but_keeps_degraded(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    for run, lt, st in (
        ("bad-lt", "INVALID", "VALID"),
        ("bad-st", "VALID", "INVALID"),
        ("degraded", "DEGRADED", "DEGRADED"),
    ):
        store.save_predictions(
            run,
            "2025-01-01T00:00:00+00:00",
            "3.1.2",
            _frame().to_dict("records"),
            lt,
            st,
            "DEGRADED",
        )
        store.db.execute(
            "UPDATE prediction_outcomes SET forward_5d_return=1,forward_3m_return=2 WHERE run_id=?",
            (run,),
        )
    store.db.commit()
    report = validation_report(store.db)
    assert report["5d"]["N"] == 2
    assert report["3m"]["N"] == 2
    store.close()


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = (
        "universe.csv",
        "price_features.csv",
        "normalized_provider.csv",
        "full_analysis.csv",
    )
    for name in names:
        (source / name).write_text("symbol\nABC\n")
    manifest = {
        "scoring_model_version": "3.1.2",
        "scoring_code_fingerprint": scoring_code_fingerprint(),
        "artifact_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in names
        },
    }
    (source / "run_manifest.json").write_text(json.dumps(manifest))
    return source, manifest


def test_exact_replay_requires_fingerprint_and_every_checksum(tmp_path):
    source, manifest = _source(tmp_path)
    assert app._verify_source(source, exact=True) == manifest
    manifest["scoring_code_fingerprint"] = "wrong"
    (source / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="implementation differs"):
        app._verify_source(source, exact=True)
    manifest["scoring_code_fingerprint"] = scoring_code_fingerprint()
    manifest["artifact_sha256"].pop("universe.csv")
    (source / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="requires artifact checksums"):
        app._verify_source(source, exact=True)
    assert app._verify_source(source, exact=False) == manifest


def test_v3_to_v4_migration_and_explicit_insert_columns(tmp_path):
    path = tmp_path / "v3.db"
    old = HistoryStore(path)
    old.db.execute("UPDATE metadata SET value='3' WHERE key='database_schema_version'")
    old.db.execute("ALTER TABLE prediction_snapshots ADD COLUMN future_extension TEXT")
    old.db.commit()
    old.close()
    store = HistoryStore(path)
    store.save_predictions("run", "now", "3.1.2", _frame().to_dict("records"))
    columns = {
        row[1] for row in store.db.execute("PRAGMA table_info(prediction_snapshots)")
    }
    assert {"lt_run_status", "st_run_status", "overall_run_status"} <= columns
    assert (
        store.db.execute("SELECT symbol FROM prediction_snapshots").fetchone()[0]
        == "ABC"
    )
    store.close()


def test_future_database_is_rejected(tmp_path):
    path = tmp_path / "future.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    db.execute("INSERT INTO metadata VALUES ('database_schema_version','6')")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="future database schema"):
        HistoryStore(path)


def test_history_store_closes_after_fatal_execute_path(tmp_path, monkeypatch):
    source, _ = _source(tmp_path)
    closed = []

    class TrackingStore(HistoryStore):
        def close(self):
            closed.append(True)
            super().close()

    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "HISTORY_DB", tmp_path / "history.db")
    monkeypatch.setattr(app, "HistoryStore", TrackingStore)
    monkeypatch.setattr(
        app, "build_analysis", lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        app.execute(source.name, rescore=True)
    assert closed == [True]

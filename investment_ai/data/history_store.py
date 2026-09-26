"""Point-in-time analyst and ranking history."""

from __future__ import annotations
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from investment_ai.config import DATABASE_SCHEMA_VERSION

SNAPSHOT_FIELDS = [
    "target_low",
    "target_mean",
    "target_median",
    "target_high",
    "strong_buy",
    "buy",
    "hold",
    "sell",
    "strong_sell",
    "positive_rating_pct",
    "rating_count",
    "eps_0q_current",
    "eps_plus_1q_current",
    "eps_0y_current",
    "eps_plus_1y_current",
    "revenue_0q_avg",
    "revenue_plus_1q_avg",
    "revenue_0y_avg",
    "revenue_plus_1y_avg",
    "forward_eps_growth",
    "forward_revenue_growth",
]

COMPONENT_FIELDS = {
    "targets": {"target_low", "target_mean", "target_median", "target_high"},
    "recommendations": {
        "strong_buy",
        "buy",
        "hold",
        "sell",
        "strong_sell",
        "positive_rating_pct",
        "rating_count",
    },
    "eps_trend": {
        field
        for field in SNAPSHOT_FIELDS
        if field.startswith("eps_") and field.endswith("_current")
    },
    "earnings_estimate": {"forward_eps_growth"},
    "revenue_estimate": {
        field for field in SNAPSHOT_FIELDS if field.startswith("revenue_")
    }
    | {"forward_revenue_growth"},
}


class HistoryStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self._schema()

    def _schema(self) -> None:
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
        )
        row = self.db.execute(
            "SELECT value FROM metadata WHERE key='database_schema_version'"
        ).fetchone()
        existing = int(row[0]) if row else 1
        if existing > DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported future database schema {existing}; application supports {DATABASE_SCHEMA_VERSION}"
            )
        columns = ", ".join(f"{field} REAL" for field in SNAPSHOT_FIELDS)
        self.db.execute(
            f"CREATE TABLE IF NOT EXISTS analyst_snapshots(snapshot_date TEXT NOT NULL,fetched_at_utc TEXT NOT NULL,symbol TEXT NOT NULL,{columns},UNIQUE(fetched_at_utc,symbol))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS analyst_observations(symbol TEXT NOT NULL,component TEXT NOT NULL,observed_at_utc TEXT NOT NULL,snapshot_date TEXT NOT NULL,field TEXT NOT NULL,value REAL NOT NULL,UNIQUE(symbol,component,observed_at_utc,field))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS ranking_history(run_id TEXT,run_timestamp TEXT,symbol TEXT,long_term_score REAL,long_term_rank INTEGER,short_term_score REAL,short_term_rank INTEGER,risk_score REAL,confidence_score REAL,short_term_setup TEXT,UNIQUE(run_id,symbol))"
        )
        if existing == 1:
            self.migrate_v1_to_v2()
            existing = 2
        if existing == 2:
            self.migrate_v2_to_v3()
            existing = 3
        if existing == 3:
            self.migrate_v3_to_v4()
            existing = 4
        if existing == 4:
            self.migrate_v4_to_v5()
        self.db.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES ('database_schema_version',?)",
            (str(DATABASE_SCHEMA_VERSION),),
        )
        self.db.commit()

    def migrate_v1_to_v2(self) -> None:
        """Add point-in-time prediction and outcome storage; legacy snapshots remain archival."""
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS prediction_snapshots(
            run_id TEXT NOT NULL,run_timestamp TEXT NOT NULL,symbol TEXT NOT NULL,
            model_version TEXT,long_term_score REAL,long_term_rank INTEGER,
            short_term_score REAL,short_term_rank INTEGER,risk_score REAL,
            confidence_score REAL,short_term_setup TEXT,price_at_prediction REAL,
            benchmark_price_at_prediction REAL,index_name TEXT,sector TEXT,
            quality_score REAL,growth_score REAL,valuation_score REAL,
            expectations_score REAL,trend_score REAL,setup_quality_score REAL,
            short_expectations_score REAL,volume_score REAL,
            PRIMARY KEY(run_id,symbol))"""
        )
        columns = ",".join(
            f"{prefix}_{horizon}_return REAL"
            for horizon in ("5d", "10d", "20d", "3m", "6m", "12m")
            for prefix in ("forward", "benchmark_forward", "excess_forward")
        )
        self.db.execute(
            f"CREATE TABLE IF NOT EXISTS prediction_outcomes(run_id TEXT NOT NULL,symbol TEXT NOT NULL,{columns},PRIMARY KEY(run_id,symbol),FOREIGN KEY(run_id,symbol) REFERENCES prediction_snapshots(run_id,symbol))"
        )

    def upsert_analyst(
        self, symbol: str, data: dict[str, Any], when: datetime | None = None
    ) -> bool:
        """Deprecated legacy wide-table writer; the product flow does not call it."""
        when = when or datetime.now(timezone.utc)
        fields = ["snapshot_date", "fetched_at_utc", "symbol"] + SNAPSHOT_FIELDS
        values = [when.date().isoformat(), when.isoformat(), symbol] + [
            data.get(field) for field in SNAPSHOT_FIELDS
        ]
        query = f"INSERT OR IGNORE INTO analyst_snapshots({','.join(fields)}) VALUES ({','.join('?' * len(fields))})"
        cursor = self.db.execute(query, values)
        self.db.commit()
        return bool(cursor.rowcount)

    def upsert_component(
        self, symbol: str, component: str, data: dict[str, Any], when: datetime
    ) -> bool:
        """Write only fields owned by one freshly fetched analyst component."""
        fields = COMPONENT_FIELDS.get(component, set())
        rows = []
        for field in fields:
            value = pd.to_numeric(data.get(field), errors="coerce")
            if pd.notna(value):
                rows.append(
                    (
                        symbol,
                        component,
                        when.isoformat(),
                        when.date().isoformat(),
                        field,
                        float(value),
                    )
                )
        before = self.db.total_changes
        self.db.executemany(
            "INSERT OR IGNORE INTO analyst_observations(symbol,component,observed_at_utc,snapshot_date,field,value) VALUES (?,?,?,?,?,?)",
            rows,
        )
        self.db.commit()
        return self.db.total_changes > before

    def historical_change(
        self,
        symbol: str,
        field: str,
        days: int,
        current: Any,
        as_of: datetime | None = None,
    ):
        if field not in SNAPSHOT_FIELDS:
            return np.nan, "HISTORY_NOT_YET_AVAILABLE"
        as_of = as_of or datetime.now(timezone.utc)
        cutoff = (as_of - timedelta(days=days)).isoformat()
        row = self.db.execute(
            "SELECT value FROM analyst_observations WHERE symbol=? AND field=? AND observed_at_utc<=? ORDER BY observed_at_utc DESC LIMIT 1",
            (symbol, field, cutoff),
        ).fetchone()
        current = pd.to_numeric(current, errors="coerce")
        if not row or not row[0] or pd.isna(current):
            return np.nan, "HISTORY_NOT_YET_AVAILABLE"
        return (current - row[0]) / abs(row[0]) * 100, "AVAILABLE"

    def add_analyst_history_features(
        self, symbol: str, row: dict[str, Any], as_of: datetime | None = None
    ) -> dict[str, Any]:
        output = dict(row)
        for target in ("mean", "median"):
            for days in (7, 30, 90):
                output[f"target_{target}_change_{days}d_pct"], _ = (
                    self.historical_change(
                        symbol,
                        f"target_{target}",
                        days,
                        row.get(f"target_{target}"),
                        as_of,
                    )
                )
        for horizon in ("0q", "plus_1q"):
            for days in (7, 30, 90):
                output[f"revenue_{horizon}_change_{days}d_pct"], _ = (
                    self.historical_change(
                        symbol,
                        f"revenue_{horizon}_avg",
                        days,
                        row.get(f"revenue_{horizon}_avg"),
                        as_of,
                    )
                )
        for horizon in ("0y", "plus_1y"):
            for days in (30, 90):
                output[f"revenue_{horizon}_change_{days}d_pct"], _ = (
                    self.historical_change(
                        symbol,
                        f"revenue_{horizon}_avg",
                        days,
                        row.get(f"revenue_{horizon}_avg"),
                        as_of,
                    )
                )
        return output

    def save_rankings(
        self, run_id: str, timestamp: str, rows: list[dict[str, Any]]
    ) -> None:
        keys = [
            "long_term_score",
            "long_term_rank",
            "short_term_score",
            "short_term_rank",
            "risk_score",
            "confidence_score",
            "short_term_setup",
        ]
        self.db.executemany(
            """INSERT OR REPLACE INTO ranking_history(
            run_id,run_timestamp,symbol,long_term_score,long_term_rank,
            short_term_score,short_term_rank,risk_score,confidence_score,short_term_setup
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [
                [run_id, timestamp, row["symbol"]] + [row.get(key) for key in keys]
                for row in rows
            ],
        )
        self.db.commit()

    def save_predictions(
        self,
        run_id: str,
        timestamp: str,
        model_version: str,
        rows: list[dict[str, Any]],
        lt_run_status: str = "VALID",
        st_run_status: str = "VALID",
        overall_run_status: str = "VALID",
    ) -> None:
        """Idempotently save scores as observed; outcome updates never recompute them."""
        keys = [
            "long_term_score",
            "long_term_rank",
            "short_term_score",
            "short_term_rank",
            "risk_score",
            "confidence_score",
            "short_term_setup",
            "current_price",
            "benchmark_price",
            "index_name",
            "sector",
            "quality_score",
            "growth_score",
            "valuation_score",
            "expectations_long_score",
            "long_trend_score",
            "setup_quality_score",
            "expectations_short_score",
            "volume_confirmation_score",
            "price_as_of",
            "benchmark_symbol",
            "benchmark_price_as_of",
        ]
        with self.db:
            self.db.executemany(
                """INSERT OR IGNORE INTO prediction_snapshots(
                run_id,run_timestamp,symbol,model_version,long_term_score,long_term_rank,
                short_term_score,short_term_rank,risk_score,confidence_score,short_term_setup,
                price_at_prediction,benchmark_price_at_prediction,index_name,sector,quality_score,
                growth_score,valuation_score,expectations_score,trend_score,setup_quality_score,
                short_expectations_score,volume_score,price_as_of_utc,benchmark_symbol,
                benchmark_price_as_of_utc,lt_run_status,st_run_status,overall_run_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    [run_id, timestamp, row["symbol"], model_version]
                    + [row.get(k) for k in keys]
                    + [lt_run_status, st_run_status, overall_run_status]
                    for row in rows
                ],
            )
            self.db.executemany(
                "INSERT OR IGNORE INTO prediction_outcomes(run_id,symbol) VALUES (?,?)",
                [(run_id, row["symbol"]) for row in rows],
            )

    def migrate_v2_to_v3(self) -> None:
        """Add immutable stock/benchmark baseline identity without rewriting snapshots."""
        columns = {
            row[1] for row in self.db.execute("PRAGMA table_info(prediction_snapshots)")
        }
        for name in (
            "price_as_of_utc",
            "benchmark_symbol",
            "benchmark_price_as_of_utc",
        ):
            if name not in columns:
                self.db.execute(
                    f"ALTER TABLE prediction_snapshots ADD COLUMN {name} TEXT"
                )

    def migrate_v3_to_v4(self) -> None:
        """Record horizon health so validation can exclude unusable predictions."""
        columns = {
            row[1] for row in self.db.execute("PRAGMA table_info(prediction_snapshots)")
        }
        for name in ("lt_run_status", "st_run_status", "overall_run_status"):
            if name not in columns:
                self.db.execute(
                    f"ALTER TABLE prediction_snapshots ADD COLUMN {name} TEXT"
                )

    def migrate_v4_to_v5(self) -> None:
        """Add stable identity, frozen benchmark metadata, outcome state, and query indexes."""
        snapshot_columns = {
            row[1] for row in self.db.execute("PRAGMA table_info(prediction_snapshots)")
        }
        additions = {
            "security_id": "TEXT",
            "benchmark_name": "TEXT",
            "benchmark_return_basis": "TEXT",
            "benchmark_currency": "TEXT",
            "benchmark_assignment_method": "TEXT",
            "trading_currency": "TEXT",
            "financial_statement_currency": "TEXT",
            "market_cap_currency": "TEXT",
        }
        for name, kind in additions.items():
            if name not in snapshot_columns:
                self.db.execute(
                    f"ALTER TABLE prediction_snapshots ADD COLUMN {name} {kind}"
                )
        self.db.execute("""CREATE TABLE IF NOT EXISTS outcome_status(
            run_id TEXT NOT NULL,symbol TEXT NOT NULL,horizon TEXT NOT NULL,
            status TEXT NOT NULL,reason TEXT,updated_at_utc TEXT NOT NULL,
            PRIMARY KEY(run_id,symbol,horizon))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS security_mappings(
            security_id TEXT NOT NULL,listing_id TEXT NOT NULL,source_index TEXT NOT NULL,
            source_symbol TEXT NOT NULL,company_name TEXT,country TEXT,exchange TEXT,isin TEXT,
            canonical_symbol TEXT,mapping_status TEXT NOT NULL,mapping_method TEXT NOT NULL,
            mapping_confidence REAL NOT NULL,verified_at_utc TEXT,last_seen_at_utc TEXT NOT NULL,
            mapping_error TEXT,PRIMARY KEY(source_index,source_symbol,country))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS daily_prices(
            security_id TEXT NOT NULL,symbol TEXT NOT NULL,date TEXT NOT NULL,
            adjusted_close REAL,raw_close REAL,volume REAL,currency TEXT,exchange TEXT,
            source TEXT NOT NULL,fetched_at_utc TEXT NOT NULL,quality_status TEXT NOT NULL,
            quality_flags TEXT,price_repair_attempted INTEGER NOT NULL DEFAULT 0,
            price_repair_succeeded INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(security_id,date))""")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyst_symbol_field_time ON analyst_observations(symbol,field,observed_at_utc)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ranking_symbol_time ON ranking_history(symbol,run_timestamp)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_time_symbol ON prediction_snapshots(run_timestamp,symbol)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_outcomes_run_symbol ON prediction_outcomes(run_id,symbol)"
        )

    def changes(self, symbol: str, days: int = 7, as_of: datetime | None = None):
        cutoff = (
            (as_of or datetime.now(timezone.utc)) - timedelta(days=days)
        ).isoformat()
        row = self.db.execute(
            "SELECT * FROM ranking_history WHERE symbol=? AND run_timestamp<=? ORDER BY run_timestamp DESC LIMIT 1",
            (symbol, cutoff),
        ).fetchone()
        return dict(row) if row else None

    def changes_by_field(
        self, symbol: str, days: int = 7, as_of: datetime | None = None
    ) -> dict[str, Any]:
        """Select the latest non-null historical value independently per field."""
        cutoff = (
            (as_of or datetime.now(timezone.utc)) - timedelta(days=days)
        ).isoformat()
        output = {}
        for field in (
            "long_term_score",
            "long_term_rank",
            "short_term_score",
            "short_term_rank",
        ):
            row = self.db.execute(
                f"SELECT {field} FROM ranking_history WHERE symbol=? AND run_timestamp<=? "
                f"AND {field} IS NOT NULL ORDER BY run_timestamp DESC LIMIT 1",
                (symbol, cutoff),
            ).fetchone()
            output[field] = row[0] if row else None
            output[f"{field}_history_status"] = (
                "AVAILABLE" if row else "NO_VALID_HISTORICAL_VALUE"
            )
        return output

    def pending_outcome_symbols(self, as_of: datetime | None = None) -> set[str]:
        """Return historical symbols with at least one matured, unpriced horizon."""
        as_of = as_of or datetime.now(timezone.utc)
        rows = self.db.execute("""SELECT p.symbol,p.run_timestamp,o.* FROM prediction_snapshots p
            JOIN prediction_outcomes o USING(run_id,symbol)""").fetchall()
        fields = [
            d[0]
            for d in self.db.execute("""SELECT p.symbol,p.run_timestamp,o.* FROM
            prediction_snapshots p JOIN prediction_outcomes o USING(run_id,symbol) LIMIT 0""").description
        ]
        horizons = {"5d": 5, "10d": 10, "20d": 20, "3m": 63, "6m": 126, "12m": 252}
        pending = set()
        for raw in rows:
            row = dict(zip(fields, raw))
            started = datetime.fromisoformat(row["run_timestamp"])
            if any(
                as_of >= started + timedelta(days=int(sessions * 7 / 5) + 4)
                and row.get(f"forward_{label}_return") is None
                for label, sessions in horizons.items()
            ):
                pending.add(row["symbol"])
        return pending

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

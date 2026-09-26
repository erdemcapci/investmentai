"""Point-in-time analyst and ranking history."""

from __future__ import annotations
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

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
    "recommendations": {"strong_buy", "buy", "hold", "sell", "strong_sell", "positive_rating_pct", "rating_count"},
    "eps_trend": {field for field in SNAPSHOT_FIELDS if field.startswith("eps_") and field.endswith("_current")},
    "earnings_estimate": {"forward_eps_growth"},
    "revenue_estimate": {field for field in SNAPSHOT_FIELDS if field.startswith("revenue_")} | {"forward_revenue_growth"},
}


class HistoryStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self._schema()

    def _schema(self) -> None:
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
        self.db.commit()

    def upsert_analyst(
        self, symbol: str, data: dict[str, Any], when: datetime | None = None
    ) -> bool:
        """Persist only a genuine provider observation timestamp; duplicates are ignored."""
        when = when or datetime.now(timezone.utc)
        fields = ["snapshot_date", "fetched_at_utc", "symbol"] + SNAPSHOT_FIELDS
        values = [when.date().isoformat(), when.isoformat(), symbol] + [
            data.get(field) for field in SNAPSHOT_FIELDS
        ]
        query = f"INSERT OR IGNORE INTO analyst_snapshots({','.join(fields)}) VALUES ({','.join('?' * len(fields))})"
        cursor = self.db.execute(query, values)
        self.db.commit()
        return bool(cursor.rowcount)

    def upsert_component(self, symbol: str, component: str, data: dict[str, Any], when: datetime) -> bool:
        """Write only fields owned by one freshly fetched analyst component."""
        fields = COMPONENT_FIELDS.get(component, set())
        rows = []
        for field in fields:
            value = pd.to_numeric(data.get(field), errors="coerce")
            if pd.notna(value):
                rows.append((symbol, component, when.isoformat(), when.date().isoformat(), field, float(value)))
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
        cutoff = (as_of.date() - timedelta(days=days)).isoformat()
        row = self.db.execute(
            "SELECT value FROM analyst_observations WHERE symbol=? AND field=? AND snapshot_date<=? ORDER BY observed_at_utc DESC LIMIT 1",
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
                output[f"revenue_{horizon}_change_{days}d_pct"], _ = self.historical_change(
                    symbol, f"revenue_{horizon}_avg", days,
                    row.get(f"revenue_{horizon}_avg"), as_of)
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
            f"INSERT OR REPLACE INTO ranking_history VALUES ({','.join('?' * 10)})",
            [
                [run_id, timestamp, row["symbol"]] + [row.get(key) for key in keys]
                for row in rows
            ],
        )
        self.db.commit()

    def changes(self, symbol: str, days: int = 7, as_of: datetime | None = None):
        cutoff = (
            (as_of or datetime.now(timezone.utc)) - timedelta(days=days)
        ).isoformat()
        row = self.db.execute(
            "SELECT * FROM ranking_history WHERE symbol=? AND run_timestamp<=? ORDER BY run_timestamp DESC LIMIT 1",
            (symbol, cutoff),
        ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self.db.close()

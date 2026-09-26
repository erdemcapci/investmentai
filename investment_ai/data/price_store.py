"""Canonical local daily-price persistence and incremental fetch planning."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import sqlite3
from typing import Iterable, Mapping

import pandas as pd


class PriceStore:
    def __init__(self, connection: sqlite3.Connection):
        self.db = connection
        self.db.execute("""CREATE TABLE IF NOT EXISTS daily_prices(
            security_id TEXT NOT NULL, symbol TEXT NOT NULL, date TEXT NOT NULL,
            adjusted_close REAL, raw_close REAL, volume REAL, currency TEXT,
            exchange TEXT, source TEXT NOT NULL, fetched_at_utc TEXT NOT NULL,
            quality_status TEXT NOT NULL, quality_flags TEXT,
            price_repair_attempted INTEGER NOT NULL DEFAULT 0,
            price_repair_succeeded INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(security_id,date))""")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol_date ON daily_prices(symbol,date)"
        )
        self.db.commit()

    def upsert(self, rows: Iterable[Mapping]) -> int:
        values = []
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            values.append(
                (
                    row.get("security_id") or row["symbol"],
                    row["symbol"],
                    str(row["date"])[:10],
                    row.get("adjusted_close"),
                    row.get("raw_close"),
                    row.get("volume"),
                    row.get("currency"),
                    row.get("exchange"),
                    row.get("source", "YAHOO"),
                    row.get("fetched_at_utc", now),
                    row.get("quality_status", "OK"),
                    row.get("quality_flags"),
                    int(row.get("price_repair_attempted", False)),
                    int(row.get("price_repair_succeeded", False)),
                )
            )
        before = self.db.total_changes
        with self.db:
            self.db.executemany(
                """INSERT INTO daily_prices VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(security_id,date) DO UPDATE SET
                symbol=excluded.symbol,adjusted_close=excluded.adjusted_close,
                raw_close=excluded.raw_close,volume=excluded.volume,currency=excluded.currency,
                exchange=excluded.exchange,source=excluded.source,fetched_at_utc=excluded.fetched_at_utc,
                quality_status=excluded.quality_status,quality_flags=excluded.quality_flags,
                price_repair_attempted=excluded.price_repair_attempted,
                price_repair_succeeded=excluded.price_repair_succeeded""",
                values,
            )
        return self.db.total_changes - before

    def history(self, security_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM daily_prices WHERE security_id=? ORDER BY date",
            self.db,
            params=(security_id,),
            parse_dates=["date"],
        )

    def histories(self, securities: Mapping[str, str]) -> pd.DataFrame:
        """Reconstruct the yfinance-shaped canonical adjusted history."""
        frames = {}
        for security_id, symbol in securities.items():
            history = self.history(security_id)
            if history.empty:
                continue
            frame = history.set_index("date")[["adjusted_close", "volume"]].rename(
                columns={"adjusted_close": "Close", "volume": "Volume"}
            )
            frames[symbol] = frame
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    def security_id_for_symbol(self, symbol: str) -> str | None:
        row = self.db.execute(
            "SELECT security_id FROM daily_prices WHERE symbol=? "
            "ORDER BY date DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return row[0] if row else None

    def missing_ranges(
        self,
        securities: Mapping[str, str],
        start: date,
        end: date,
        overlap_days: int = 5,
    ) -> dict[str, tuple[date, date]]:
        """Return only uncached/recent ranges, with a small revision overlap."""
        output = {}
        for security_id in securities:
            row = self.db.execute(
                "SELECT MAX(date) FROM daily_prices WHERE security_id=?", (security_id,)
            ).fetchone()
            latest = date.fromisoformat(row[0]) if row and row[0] else None
            fetch_start = (
                start
                if latest is None
                else max(start, latest - timedelta(days=overlap_days))
            )
            if fetch_start <= end:
                output[security_id] = (fetch_start, end)
        return output


def price_quality(
    frame: pd.DataFrame, extreme_move: float = 0.60
) -> tuple[str, list[str]]:
    """Flag, rather than silently repair, suspicious daily histories."""
    flags = []
    if frame.empty or "Close" not in frame:
        return "INSUFFICIENT", ["NO_CLOSE_HISTORY"]
    close = pd.to_numeric(frame["Close"], errors="coerce")
    if close.le(0).any():
        flags.append("NON_POSITIVE_CLOSE")
    moves = close.pct_change().abs()
    if moves.gt(extreme_move).any():
        flags.append("EXTREME_DAILY_MOVE")
    ratios = close / close.shift(1)
    if (ratios.ge(90) | ratios.le(1 / 90)).any():
        flags.append("UNIT_JUMP")
    if frame.index.duplicated().any():
        flags.append("DUPLICATE_TIMESTAMP")
    if {"High", "Low"} <= set(frame):
        high, low = (
            pd.to_numeric(frame.High, errors="coerce"),
            pd.to_numeric(frame.Low, errors="coerce"),
        )
        if high.lt(low).any():
            flags.append("IMPOSSIBLE_OHLC")
    return ("SUSPECT" if flags else "OK"), flags

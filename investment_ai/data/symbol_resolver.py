"""Durable, auditable constituent-to-vendor symbol resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import re
import sqlite3
from typing import Callable, Mapping, Any

from investment_ai.data.constituents import YAHOO_SUFFIX_BY_COUNTRY

STATUSES = {"VERIFIED", "HEURISTIC", "UNRESOLVED", "AMBIGUOUS", "STALE_MAPPING"}
KNOWN_EXCEPTIONS = {
    ("sweden", "ATCOA"): "ATCO-A.ST",
    ("sweden", "ERICB"): "ERIC-B.ST",
    ("sweden", "HMB"): "HM-B.ST",
    ("switzerland", "SRENH"): "SREN.SW",
}


@dataclass(frozen=True)
class SymbolMapping:
    security_id: str
    listing_id: str
    source_index: str
    source_symbol: str
    company_name: str
    country: str
    exchange: str | None
    isin: str | None
    canonical_yahoo_symbol: str | None
    mapping_method: str
    mapping_status: str
    mapping_confidence: float
    mapping_verified_at_utc: str | None
    mapping_error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _identity(
    source_index: str, source_symbol: str, company: str, country: str, isin: str | None
) -> str:
    if isin and str(isin).strip():
        return "ISIN:" + str(isin).strip().upper()
    stable = "|".join(
        map(
            lambda x: str(x).strip().casefold(),
            (source_index, source_symbol, company, country),
        )
    )
    return "SEC:" + hashlib.sha256(stable.encode()).hexdigest()[:24]


def _candidate(symbol: str, country: str) -> tuple[str | None, str]:
    raw = re.sub(r"\s+", "", str(symbol)).upper()
    key = (country.strip().lower(), raw.replace(".", ""))
    if key in KNOWN_EXCEPTIONS:
        return KNOWN_EXCEPTIONS[key], "KNOWN_VENDOR_EXCEPTION"
    suffix = YAHOO_SUFFIX_BY_COUNTRY.get(country.strip().lower())
    if not raw or raw == "NAN" or not suffix:
        return None, "UNSUPPORTED_OR_MISSING"
    if raw.endswith(suffix):
        return raw, "VENDOR_QUALIFIED"
    # Only transformations with unambiguous local syntax are deterministic.
    return raw.replace(".", "-") + suffix, "EXCHANGE_SUFFIX_HEURISTIC"


class SymbolResolver:
    """Resolve using persisted verification, known rules, then optional metadata."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        metadata_lookup: Callable[
            [str], Mapping[str, Any] | list[Mapping[str, Any]] | None
        ]
        | None = None,
    ):
        self.db = connection
        self.metadata_lookup = metadata_lookup
        self._schema()

    def _schema(self) -> None:
        self.db.execute("""CREATE TABLE IF NOT EXISTS security_mappings(
            security_id TEXT NOT NULL, listing_id TEXT NOT NULL, source_index TEXT NOT NULL,
            source_symbol TEXT NOT NULL, company_name TEXT, country TEXT, exchange TEXT, isin TEXT,
            canonical_symbol TEXT, mapping_status TEXT NOT NULL, mapping_method TEXT NOT NULL,
            mapping_confidence REAL NOT NULL, verified_at_utc TEXT, last_seen_at_utc TEXT NOT NULL,
            mapping_error TEXT, PRIMARY KEY(source_index,source_symbol,country))""")
        self.db.commit()

    def resolve(
        self,
        source_index: str,
        source_symbol: str,
        company_name: str,
        country: str,
        exchange: str | None = None,
        isin: str | None = None,
    ) -> SymbolMapping:
        now = datetime.now(timezone.utc).isoformat()
        persisted = self.db.execute(
            "SELECT * FROM security_mappings WHERE source_index=? AND source_symbol=? AND country=?",
            (source_index, source_symbol, country),
        ).fetchone()
        if persisted and persisted[9] == "VERIFIED":
            values = dict(
                zip(
                    [
                        d[0]
                        for d in self.db.execute(
                            "SELECT * FROM security_mappings LIMIT 0"
                        ).description
                    ],
                    persisted,
                )
            )
            self.db.execute(
                "UPDATE security_mappings SET last_seen_at_utc=? WHERE source_index=? AND source_symbol=? AND country=?",
                (now, source_index, source_symbol, country),
            )
            self.db.commit()
            return SymbolMapping(
                values["security_id"],
                values["listing_id"],
                source_index,
                source_symbol,
                company_name,
                country,
                exchange,
                isin,
                values["canonical_symbol"],
                "PERSISTED_VERIFIED",
                "VERIFIED",
                values["mapping_confidence"],
                values["verified_at_utc"],
            )
        candidate, method = _candidate(source_symbol, country)
        status, confidence, verified, error = "UNRESOLVED", 0.0, None, None
        if candidate:
            status, confidence = (
                ("VERIFIED", 0.99)
                if method in {"KNOWN_VENDOR_EXCEPTION", "VENDOR_QUALIFIED"}
                else ("HEURISTIC", 0.65)
            )
            if self.metadata_lookup:
                metadata = self.metadata_lookup(candidate)
                if isinstance(metadata, list):
                    status, error = "AMBIGUOUS", "multiple provider candidates"
                elif metadata:
                    expected = set(re.findall(r"[a-z0-9]+", company_name.casefold()))
                    actual = set(
                        re.findall(
                            r"[a-z0-9]+", str(metadata.get("name", "")).casefold()
                        )
                    )
                    country_ok = (
                        not metadata.get("country")
                        or str(metadata["country"]).casefold() == country.casefold()
                    )
                    exchange_ok = (
                        not exchange
                        or not metadata.get("exchange")
                        or str(metadata["exchange"]).casefold() == exchange.casefold()
                    )
                    if expected & actual and country_ok and exchange_ok:
                        status, confidence, verified, method = (
                            "VERIFIED",
                            0.95,
                            now,
                            "PROVIDER_METADATA",
                        )
                    else:
                        status, error = "AMBIGUOUS", "provider identity mismatch"
        else:
            error = method
        security_id = _identity(
            source_index, source_symbol, company_name, country, isin
        )
        listing_id = hashlib.sha256(
            f"{security_id}|{exchange or ''}|{candidate or ''}".encode()
        ).hexdigest()[:24]
        result = SymbolMapping(
            security_id,
            listing_id,
            source_index,
            source_symbol,
            company_name,
            country,
            exchange,
            isin,
            candidate,
            method,
            status,
            confidence,
            verified,
            error,
        )
        self.db.execute(
            """INSERT OR REPLACE INTO security_mappings VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                security_id,
                listing_id,
                source_index,
                source_symbol,
                company_name,
                country,
                exchange,
                isin,
                candidate,
                status,
                method,
                confidence,
                verified,
                now,
                error,
            ),
        )
        self.db.commit()
        return result

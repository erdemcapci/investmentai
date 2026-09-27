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
COMPANY_NOISE = {"plc", "ag", "sa", "se", "nv", "ab", "group", "holding", "holdings", "the"}
NORDIC_CLASS_COUNTRIES = {"sweden", "denmark", "finland", "norway"}
YAHOO_EXCHANGE_BY_COUNTRY = {
    "austria": {"VIE", "VIENNA"},
    "belgium": {"BRU", "BRUSSELS"},
    "denmark": {"CPH", "COPENHAGEN"},
    "finland": {"HEL", "HELSINKI"},
    "france": {"PAR", "PARIS"},
    "germany": {"GER", "XETRA"},
    "ireland": {"ISE", "IRISH"},
    "italy": {"MIL", "MILAN"},
    "netherlands": {"AMS", "AMSTERDAM"},
    "norway": {"OSL", "OSLO"},
    "poland": {"WSE", "WARSAW"},
    "portugal": {"LIS", "LISBON"},
    "spain": {"MCE", "MADRID"},
    "sweden": {"STO", "STOCKHOLM"},
    "switzerland": {"EBS", "SWISS"},
    "united kingdom": {"LSE", "LONDON"},
    "uk": {"LSE", "LONDON"},
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


def _company_declares_share_class(company_name: str, share_class: str) -> bool:
    name = str(company_name).strip().upper()
    patterns = (
        rf"\bCLASS\s+{share_class}\b",
        rf"[- ]{share_class}\s+(?:SHS|SHARES)\b",
        rf"\b{share_class}$",
    )
    return any(re.search(pattern, name) for pattern in patterns)


def normalize_local_ticker(
    source_symbol: str,
    country: str | None,
    exchange: str | None = None,
    company_name: str = "",
) -> tuple[str, str]:
    """Normalize only deterministic exchange-specific source conventions."""
    raw = str(source_symbol).strip().upper()
    country_key = str(country or "").strip().casefold()
    exchange_key = str(exchange or "").strip().casefold()

    uk_context = country_key in {"united kingdom", "uk"} or "london" in exchange_key
    if uk_context and re.fullmatch(r"[A-Z0-9][A-Z0-9./-]*[.-]", raw):
        return raw[:-1], "UK_TRAILING_SEPARATOR"
    if uk_context:
        uk_class = re.fullmatch(r"([A-Z0-9]+)\.([A-Z])", raw)
        if uk_class:
            return f"{uk_class.group(1)}-{uk_class.group(2)}", "UK_SHARE_CLASS"

    nordic_context = country_key in NORDIC_CLASS_COUNTRIES or any(
        marker in exchange_key
        for marker in ("stockholm", "copenhagen", "helsinki", "oslo")
    )
    if nordic_context:
        finnish_listing = re.fullmatch(r"([A-Z0-9]+)[ .-](FI)", raw)
        if country_key == "finland" and finnish_listing:
            return (
                f"{finnish_listing.group(1)}-{finnish_listing.group(2)}",
                "FINNISH_LISTING_MARKER",
            )
        explicit = re.fullmatch(r"([A-Z0-9]+)[ .-](A|B|C|SDB)", raw)
        if explicit:
            return f"{explicit.group(1)}-{explicit.group(2)}", "NORDIC_SHARE_CLASS"
        compressed = re.fullmatch(r"([A-Z0-9]+)([AB])", raw)
        if compressed and _company_declares_share_class(
            company_name, compressed.group(2)
        ):
            return (
                f"{compressed.group(1)}-{compressed.group(2)}",
                "NORDIC_SHARE_CLASS",
            )

    return raw, "LOCAL_TICKER_UNCHANGED"


def _exchange_matches(
    source_exchange: str | None,
    provider_exchange: str | None,
    provider_full_exchange: str | None,
    country: str,
) -> bool:
    if not source_exchange or (not provider_exchange and not provider_full_exchange):
        return True
    source = str(source_exchange).strip().casefold()
    provider_values = {
        str(value).strip().upper()
        for value in (provider_exchange, provider_full_exchange)
        if value
    }
    if source in {value.casefold() for value in provider_values}:
        return True
    expected = YAHOO_EXCHANGE_BY_COUNTRY.get(str(country).strip().casefold(), set())
    return bool(provider_values & expected)


def _candidate(
    symbol: str,
    country: str,
    exchange: str | None = None,
    company_name: str = "",
) -> tuple[str | None, str]:
    raw = str(symbol).strip().upper()
    country_key = str(country or "").strip().lower()
    suffix = YAHOO_SUFFIX_BY_COUNTRY.get(country_key)
    if raw and any(raw.endswith(value) for value in set(YAHOO_SUFFIX_BY_COUNTRY.values())):
        return raw, "VENDOR_QUALIFIED"
    compact = re.sub(r"\s+", "", raw)
    key = (country_key, compact.replace(".", ""))
    if key in KNOWN_EXCEPTIONS:
        return KNOWN_EXCEPTIONS[key], "KNOWN_VENDOR_EXCEPTION"
    local, normalization_method = normalize_local_ticker(
        raw, country, exchange, company_name
    )
    if not local or local == "NAN" or not suffix:
        return None, "UNSUPPORTED_OR_MISSING"
    if re.search(r"\s", local):
        return None, "AMBIGUOUS_LOCAL_FORMAT"
    method = (
        normalization_method
        if normalization_method != "LOCAL_TICKER_UNCHANGED"
        else "EXCHANGE_SUFFIX_HEURISTIC"
    )
    return local + suffix, method


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
        candidate, method = _candidate(
            source_symbol, country, exchange, company_name
        )
        persisted = self.db.execute(
            "SELECT * FROM security_mappings WHERE source_index=? AND source_symbol=? AND country=?",
            (source_index, source_symbol, country),
        ).fetchone()
        persisted_invalid = bool(
            persisted
            and persisted[9] == "UNRESOLVED"
            and persisted[10] == "YAHOO_INVALID_SYMBOL"
            and persisted[8] == candidate
        )
        if persisted and (persisted[9] == "VERIFIED" or persisted_invalid):
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
                "PERSISTED_VERIFIED" if persisted[9] == "VERIFIED" else values["mapping_method"],
                values["mapping_status"],
                values["mapping_confidence"],
                values["verified_at_utc"],
                values["mapping_error"],
            )
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
                    expected = set(re.findall(r"[a-z0-9]+", company_name.casefold())) - COMPANY_NOISE
                    provider_name = metadata.get("longName") or metadata.get("shortName") or metadata.get("name", "")
                    actual = set(
                        re.findall(r"[a-z0-9]+", str(provider_name).casefold())
                    ) - COMPANY_NOISE
                    country_ok = (
                        not metadata.get("country")
                        or str(metadata["country"]).casefold() == country.casefold()
                    )
                    exchange_ok = _exchange_matches(
                        exchange,
                        metadata.get("exchange"),
                        metadata.get("fullExchangeName"),
                        country,
                    )
                    quote_type = str(metadata.get("quoteType", "")).upper()
                    listing_ok = not quote_type or quote_type == "EQUITY"
                    # A meaningful shared identity token is required; merely
                    # receiving a Yahoo object never verifies a mapping.
                    name_ok = bool(expected and actual and expected & actual)
                    if name_ok and country_ok and exchange_ok and listing_ok:
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

    def mark_invalid_yahoo(
        self, mapping: Mapping[str, Any], error: str
    ) -> SymbolMapping:
        """Persist a definitive provider rejection so later runs make no calls."""
        current = SymbolMapping(
            **{
                field: mapping.get(field)
                for field in SymbolMapping.__dataclass_fields__
            }
        )
        now = datetime.now(timezone.utc).isoformat()
        result = SymbolMapping(
            current.security_id,
            current.listing_id,
            current.source_index,
            current.source_symbol,
            current.company_name,
            current.country,
            current.exchange,
            current.isin,
            current.canonical_yahoo_symbol,
            "YAHOO_INVALID_SYMBOL",
            "UNRESOLVED",
            0.0,
            None,
            str(error)[:1000],
        )
        self.db.execute(
            """UPDATE security_mappings SET mapping_status=?,mapping_method=?,
            mapping_confidence=?,verified_at_utc=?,last_seen_at_utc=?,mapping_error=?
            WHERE source_index=? AND source_symbol=? AND country=?""",
            (
                result.mapping_status,
                result.mapping_method,
                result.mapping_confidence,
                result.mapping_verified_at_utc,
                now,
                result.mapping_error,
                result.source_index,
                result.source_symbol,
                result.country,
            ),
        )
        self.db.commit()
        return result

    def verify(self, mapping: Mapping[str, Any], metadata: Mapping[str, Any] | None) -> SymbolMapping:
        """Verify a deterministic candidate using metadata captured by the provider pass."""
        current = SymbolMapping(**{field: mapping.get(field) for field in SymbolMapping.__dataclass_fields__})
        if current.mapping_status == "VERIFIED" or not current.canonical_yahoo_symbol:
            return current
        if not metadata:
            return current
        now = datetime.now(timezone.utc).isoformat()
        expected = set(re.findall(r"[a-z0-9]+", current.company_name.casefold())) - COMPANY_NOISE
        provider_name = metadata.get("longName") or metadata.get("shortName") or metadata.get("name", "")
        actual = set(re.findall(r"[a-z0-9]+", str(provider_name).casefold())) - COMPANY_NOISE
        country_ok = not metadata.get("country") or str(metadata["country"]).casefold() == current.country.casefold()
        exchange_ok = _exchange_matches(
            current.exchange,
            metadata.get("exchange"),
            metadata.get("fullExchangeName"),
            current.country,
        )
        quote_type = str(metadata.get("quoteType", "")).upper()
        provider_symbol = str(metadata.get("symbol", "")).upper()
        symbol_ok = not provider_symbol or provider_symbol == current.canonical_yahoo_symbol.upper()
        identity_ok = bool(expected and actual and expected & actual) and country_ok and exchange_ok and symbol_ok and (not quote_type or quote_type == "EQUITY")
        status = "VERIFIED" if identity_ok else "AMBIGUOUS"
        result = SymbolMapping(
            current.security_id, current.listing_id, current.source_index,
            current.source_symbol, current.company_name, current.country,
            current.exchange, current.isin, current.canonical_yahoo_symbol,
            "PROVIDER_METADATA" if identity_ok else current.mapping_method,
            status, 0.95 if identity_ok else current.mapping_confidence,
            now if identity_ok else None,
            None if identity_ok else "provider identity mismatch",
        )
        self.db.execute(
            """UPDATE security_mappings SET mapping_status=?,mapping_method=?,
            mapping_confidence=?,verified_at_utc=?,last_seen_at_utc=?,mapping_error=?
            WHERE source_index=? AND source_symbol=? AND country=?""",
            (result.mapping_status, result.mapping_method, result.mapping_confidence,
             result.mapping_verified_at_utc, now, result.mapping_error,
             result.source_index, result.source_symbol, result.country),
        )
        self.db.commit()
        return result

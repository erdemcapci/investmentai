"""Investment AI v1.1 entry point. Canonical usage: ``python main.py``."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import uuid
import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from investment_ai.config import (
    APPLICATION_VERSION,
    CACHE_DIR,
    EXPORT_RESULTS,
    FORCE_REFRESH,
    HISTORY_DB,
    MAX_WORKERS,
    MIN_PEERS,
    PRICE_BATCH_SIZE,
    PRICE_PERIOD,
    RUNS_DIR,
    SCORING_MODEL_VERSION,
    TOP_N,
    validate_config,
)
from investment_ai.data.cache import JsonCache
from investment_ai.data.constituents import fetch_index_constituents
from investment_ai.data.history_store import HistoryStore
from investment_ai.data.prices import (
    build_price_features,
    download_prices,  # noqa: F401 - retained for integration monkeypatch compatibility
    download_prices_range,
    symbol_history,
)
from investment_ai.data.symbol_resolver import SymbolResolver
from investment_ai.data.price_store import PriceStore, price_quality
from investment_ai.data.yahoo import YahooClient
from investment_ai.health import constituent_source_health
from investment_ai.features.benchmark import BENCHMARKS, attach_benchmark_prices
from investment_ai.pipeline import build_analysis
from investment_ai.reporting.export import export_run
from investment_ai.reporting.tables import print_rankings
from investment_ai.runtime import (
    RunContext,
    atomic_csv,
    build_manifest,
    configure_logging,
    horizon_run_statuses,
    scoring_code_fingerprint,
    scoring_parameter_snapshot,
    atomic_json,
)
from investment_ai.status import (
    ERROR,
    FRESH_CACHE,
    FRESH_PROVIDER,
    PARTIAL,
    STALE_FALLBACK,
)


def download_combined_constituents() -> pd.DataFrame:
    frame = fetch_index_constituents(CACHE_DIR / "constituents")
    present = {
        item.strip()
        for value in frame.index_name.dropna()
        for item in str(value).split("|")
    }
    if not {"S&P 500", "STOXX Europe 600"} <= present:
        raise RuntimeError(
            "Both constituent sources are required for the combined universe"
        )
    connection = sqlite3.connect(HISTORY_DB)
    # Resolution is deliberately offline.  Heuristic candidates are verified
    # later from the provider_info payload fetched by the normal provider pass.
    resolver = SymbolResolver(connection)
    resolved = []
    for row in frame.to_dict("records"):
        def present(name: str, fallback: str = ""):
            value = row.get(name)
            return value if pd.notna(value) and str(value).strip() else fallback

        source_index = str(row.get("index_name", ""))
        source_value = next(
            (
                value
                for value in (
                    row.get("source_symbol"),
                    row.get("stoxx_ticker"),
                    row.get("symbol"),
                )
                if pd.notna(value) and str(value).strip()
            ),
            "",
        )
        source_symbol = str(source_value)
        if "S&P 500" in source_index:
            # US source symbols have a deterministic Yahoo class-share convention.
            mapping = resolver.resolve(
                source_index,
                row["symbol"],
                present("company_name", present("security")),
                "United States",
                present("exchange") or None,
                present("isin") or None,
            )
            data = mapping.as_dict()
            data.update(
                {
                    "canonical_yahoo_symbol": row["symbol"],
                    "mapping_status": "VERIFIED",
                    "mapping_method": "SP500_DIRECT",
                }
            )
        else:
            data = resolver.resolve(
                source_index,
                source_symbol,
                present("company_name", present("security")),
                present("country"),
                present("exchange") or None,
                present("isin") or None,
            ).as_dict()
        row.update(data)
        row["symbol"] = data.get("canonical_yahoo_symbol") or row.get("symbol")
        resolved.append(row)
    connection.close()
    return pd.DataFrame(resolved)


def _constituent_health(universe: pd.DataFrame, now: datetime) -> dict[str, dict]:
    output = {}
    memberships = universe.get("index_name", pd.Series("", index=universe.index)).astype(str)
    for key, name in (("sp500", "S&P 500"), ("stoxx600", "STOXX Europe 600")):
        source = universe[memberships.str.contains(name, regex=False)]
        statuses = source.get("mapping_status", pd.Series("VERIFIED", index=source.index))
        timestamps = source.get(
            "constituent_list_fetched_at_utc", pd.Series(pd.NaT, index=source.index)
        )
        fetched = pd.to_datetime(
            timestamps, utc=True, errors="coerce"
        ).dropna()
        # Legacy frozen artifacts have no age telemetry; do not invent staleness.
        age = max(0.0, (now - fetched.min().to_pydatetime()).total_seconds() / 3600) if len(fetched) else 0.0
        verified = int(statuses.eq("VERIFIED").sum())
        usable = int(statuses.isin(["VERIFIED", "HEURISTIC"]).sum())
        health = constituent_source_health(
            name, len(source), usable, age,
            bool(source.get("constituent_source_cache_used", pd.Series(False, index=source.index)).fillna(False).astype(bool).any()),
        )
        health.update({
            "raw_count": len(source),
            "verified_mapping_count": verified,
            "heuristic_mapping_count": int(statuses.eq("HEURISTIC").sum()),
            "unresolved_count": int(statuses.eq("UNRESOLVED").sum()),
            "ambiguous_count": int(statuses.eq("AMBIGUOUS").sum()),
            "usable_mapping_pct": float(statuses.isin(["VERIFIED", "HEURISTIC"]).mean() * 100) if len(source) else 0.0,
            "status": health["source_status"],
        })
        # Verification unavailability is visible but is not an integrity
        # failure when a deterministic candidate remains otherwise usable.
        if health["source_status"] == "VALID" and statuses.eq("HEURISTIC").any():
            health["source_status"] = health["status"] = "DEGRADED"
            health["status_reasons"].append("provider mapping verification unavailable")
        output[key] = health
    return output


def _verify_provider_mappings(universe: pd.DataFrame, provider: pd.DataFrame) -> pd.DataFrame:
    """Apply shared provider_info metadata without making another Yahoo request."""
    result = universe.copy()
    if result.empty or provider.empty or "mapping_status" not in result:
        return result
    metadata_fields = YahooClient.INFO_FIELDS
    by_symbol = provider.drop_duplicates("symbol", keep="last").set_index("symbol")
    connection = sqlite3.connect(HISTORY_DB)
    resolver = SymbolResolver(connection)
    try:
        for index, row in result[result.mapping_status.eq("HEURISTIC")].iterrows():
            if row.symbol not in by_symbol.index:
                continue
            captured = by_symbol.loc[row.symbol]
            if not bool(captured.get("info_success", False)):
                continue
            metadata = {
                field: captured.get(f"provider_info_{field}") for field in metadata_fields
                if pd.notna(captured.get(f"provider_info_{field}"))
            }
            if not metadata:
                continue
            mapping = resolver.verify(row.to_dict(), metadata)
            for key, value in mapping.as_dict().items():
                result.at[index, key] = value
    finally:
        connection.close()
    return result


def _incremental_price_history(
    price_store: PriceStore,
    securities: dict[str, str],
    start: date,
    end: date,
) -> tuple[pd.DataFrame, int]:
    """Fetch grouped missing windows, persist deltas, then reload canonical history."""
    ranges = price_store.missing_ranges(securities, start, end)
    groups: dict[tuple[date, date], list[tuple[str, str]]] = {}
    for security_id, window in ranges.items():
        groups.setdefault(window, []).append((security_id, securities[security_id]))
    fetched_symbols = 0
    for (fetch_start, fetch_end), members in groups.items():
        symbols = [symbol for _, symbol in members]
        delta = download_prices_range(symbols, fetch_start, fetch_end + timedelta(days=1))
        fetched_symbols += len(symbols)
        ids = {symbol: security_id for security_id, symbol in members}
        rows = []
        for symbol in symbols:
            history = symbol_history(delta, symbol)
            quality, flags = price_quality(history)
            for day, bar in history.iterrows():
                rows.append({
                    "security_id": ids[symbol], "symbol": symbol, "date": day,
                    "adjusted_close": bar.get("Close"), "raw_close": None,
                    "volume": bar.get("Volume"), "source": "YAHOO",
                    "quality_status": quality, "quality_flags": "|".join(flags),
                })
        price_store.upsert(rows)
    return price_store.histories(securities), fetched_symbols


def methodology() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "score": "Long term",
                "formula": "25% Quality + 20% Growth + 20% Valuation + 20% Long-Term Expectations + 10% Long Trend + 5% Financial Safety",
            },
            {
                "score": "Short term",
                "formula": "20% raw Relative Strength + 25% Setup + 20% Short-Term Expectations + 10% Volume + 15% Technical Trend + 10% Event Timing",
            },
            {
                "score": "Benchmark RS",
                "formula": "stock return - assigned regional benchmark return; experimental and retained beside raw RS",
            },
        ]
    )


def _rank_changes(
    frame: pd.DataFrame, store: HistoryStore, started: datetime
) -> pd.DataFrame:
    result = frame.copy()
    values = []
    statuses = []
    for row in result.to_dict("records"):
        old = store.changes_by_field(row["symbol"], 7, started)
        if not old:
            values.append((np.nan, np.nan, np.nan, np.nan, "NEW"))
            statuses.append(("HISTORY_NOT_YET_AVAILABLE",) * 4)
        else:
            changes = []
            for key, rank in (
                ("long_term_score", False),
                ("short_term_score", False),
                ("long_term_rank", True),
                ("short_term_rank", True),
            ):
                current, previous = row.get(key), old[key]
                changes.append(
                    (previous - current if rank else current - previous)
                    if pd.notna(current) and previous is not None
                    else np.nan
                )
            values.append(tuple(changes) + ("AVAILABLE",))
            statuses.append(
                tuple(
                    old[f"{key}_history_status"]
                    for key in (
                        "long_term_score",
                        "short_term_score",
                        "long_term_rank",
                        "short_term_rank",
                    )
                )
            )
    result[
        [
            "long_term_score_change_7d",
            "short_term_score_change_7d",
            "long_term_rank_change_7d",
            "short_term_rank_change_7d",
            "rank_history_status",
        ]
    ] = values
    result[
        [
            "long_term_score_history_status",
            "short_term_score_history_status",
            "long_term_rank_history_status",
            "short_term_rank_history_status",
        ]
    ] = statuses
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic investment research ranking"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", metavar="RUN_ID", default=os.getenv("RESUME_RUN_ID"))
    modes.add_argument("--replay", metavar="RUN_ID")
    modes.add_argument("--rescore", metavar="RUN_ID")
    modes.add_argument("--validation-report", action="store_true")
    return parser


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=True)


def _provider_data(context: RunContext, universe: pd.DataFrame, logger) -> pd.DataFrame:
    path = context.directory / "normalized_provider.csv"
    prior = _read(path) if path.exists() else pd.DataFrame()
    artifact_symbols = (
        set(prior.symbol.astype(str)) if len(prior) and "symbol" in prior else set()
    )
    # The artifact is the durable truth; a checkpoint cannot claim an absent row.
    captured = artifact_symbols
    prior_success = pd.to_numeric(
        prior.get("provider_success", pd.Series(0, index=prior.index)), errors="coerce"
    ).gt(0)
    usable = (
        set(prior.loc[prior_success, "symbol"].astype(str)) if len(prior) else set()
    )
    context.checkpoint["provider_symbols_complete"] = sorted(
        captured
    )  # legacy resume key
    context.checkpoint["provider_symbols_usable"] = sorted(usable)
    # A durable failure row is diagnostic, not a completed capture.  A resumed
    # execution retries it once in this normal pass while skipping usable rows.
    pending = universe[~universe.symbol.isin(usable)]
    total_batches = (len(pending) + 24) // 25
    for offset in range(0, len(pending), 25):
        batch = pending.iloc[offset : offset + 25]
        batch_number = offset // 25 + 1
        logger.info(
            "provider batch %d/%d size=%d remaining=%d",
            batch_number,
            total_batches,
            len(batch),
            max(0, len(pending) - offset - len(batch)),
        )
        fresh = YahooClient(JsonCache(CACHE_DIR)).fetch_many(batch)
        prior = pd.concat(
            [prior[~prior.symbol.isin(fresh.symbol)] if len(prior) else prior, fresh],
            ignore_index=True,
        )
        success = pd.to_numeric(
            fresh.get("provider_success", pd.Series(1, index=fresh.index)),
            errors="coerce",
        ).gt(0)
        usable_now = set(fresh.loc[success, "symbol"].tolist())
        captured_now = set(fresh.symbol.tolist())
        captured.update(captured_now)
        usable.update(usable_now)
        context.checkpoint["provider_symbols_complete"] = sorted(captured)
        context.checkpoint["provider_symbols_usable"] = sorted(usable)
        context.checkpoint["provider_symbols_failed"] = sorted(captured - usable)
        atomic_csv(prior, path)
        context.save_checkpoint()
    context.checkpoint["provider_capture_complete"] = set(universe.symbol) <= captured
    context.save_checkpoint()
    return prior


def _coverage(frame: pd.DataFrame, column: str) -> float:
    return (
        float(
            frame.get(column, pd.Series(np.nan, index=frame.index)).notna().mean() * 100
        )
        if len(frame)
        else 0.0
    )


def _verify_source(source: Path, exact: bool) -> dict:
    required = (
        "run_manifest.json",
        "universe.csv",
        "price_features.csv",
        "normalized_provider.csv",
        "full_analysis.csv",
    )
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise RuntimeError(
            f"Source run is missing required artifacts: {', '.join(missing)}"
        )
    manifest = json.loads((source / "run_manifest.json").read_text())
    if exact and manifest.get("scoring_model_version") != SCORING_MODEL_VERSION:
        raise RuntimeError(
            "Exact replay requires the current scoring model version; use --rescore"
        )
    if exact:
        fingerprint = manifest.get("scoring_code_fingerprint")
        if not fingerprint:
            raise RuntimeError(
                "Legacy source has no scoring fingerprint; use --rescore"
            )
        if fingerprint != scoring_code_fingerprint():
            raise RuntimeError(
                "Exact replay scoring implementation differs; use --rescore"
            )
        parameters = manifest.get("scoring_parameter_snapshot")
        if parameters is not None and parameters != scoring_parameter_snapshot():
            raise RuntimeError("Exact replay scoring parameters differ; use --rescore")
        checksums = manifest.get("artifact_sha256", {})
        missing_hashes = [name for name in required[1:] if not checksums.get(name)]
        if missing_hashes:
            raise RuntimeError(
                "Exact replay requires artifact checksums for: "
                + ", ".join(missing_hashes)
                + "; use --rescore"
            )
        for name in required[1:]:
            digest = checksums[name]
            actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
            if actual != digest:
                raise RuntimeError(f"Exact replay artifact checksum mismatch: {name}")
    return manifest


REPLAY_COLUMNS = (
    "long_term_score",
    "long_term_rank",
    "short_term_score",
    "short_term_rank",
    "quality_score",
    "growth_score",
    "valuation_score",
    "expectations_long_score",
    "long_trend_score",
    "financial_safety_score",
    "short_rs_score",
    "setup_quality_score",
    "expectations_short_score",
    "volume_confirmation_score",
    "technical_trend_score",
    "event_timing_score",
    "risk_score",
    "confidence_score",
    "short_term_setup",
)


def verify_replay_output(
    original: pd.DataFrame, recomputed: pd.DataFrame, tolerance: float = 1e-9
) -> dict:
    left, right = original.set_index("symbol"), recomputed.set_index("symbol")
    mismatches, maximum = 0, 0.0
    if set(left.index) != set(right.index):
        mismatches += len(set(left.index) ^ set(right.index))
    for column in REPLAY_COLUMNS:
        if column not in left or column not in right:
            continue
        a, b = left[column].reindex(left.index), right[column].reindex(left.index)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            delta = (
                pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")
            ).abs()
            maximum = max(maximum, float(delta.max()) if delta.notna().any() else 0.0)
            mismatches += int((delta.gt(tolerance) | (a.isna() != b.isna())).sum())
        else:
            mismatches += int(
                (a.fillna("<NA>").astype(str) != b.fillna("<NA>").astype(str)).sum()
            )
    return {
        "replay_verification_status": "VERIFIED"
        if not mismatches
        else "REPLAY_MISMATCH",
        "max_abs_numeric_difference": maximum,
        "mismatch_count": mismatches,
    }


def _persistence_rows(full: pd.DataFrame, health: dict[str, str]) -> list[dict]:
    """Return a health-gated copy; diagnostic exports retain the original scores."""
    if health["lt_run_status"] == health["st_run_status"] == "INVALID":
        return []
    persisted = full.copy()
    if health["lt_run_status"] == "INVALID":
        persisted[["long_term_score", "long_term_rank"]] = np.nan
    if health["st_run_status"] == "INVALID":
        persisted[["short_term_score", "short_term_rank"]] = np.nan
        if "short_term_setup" in persisted:
            persisted["short_term_setup"] = None
    return persisted[
        persisted.long_term_rank.notna() | persisted.short_term_rank.notna()
    ].to_dict("records")


def execute(
    run_id: str | None = None,
    resume: bool = False,
    replay: bool = False,
    rescore: bool = False,
) -> int:
    validate_config()
    frozen_mode = replay or rescore
    source_manifest = None
    if frozen_mode:
        source = RUNS_DIR / str(run_id)
        if not source.exists():
            raise FileNotFoundError(f"Replay run not found: {run_id}")
        source_manifest = _verify_source(source, exact=replay)
        suffix = "replay" if replay else "rescore"
        child_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        context = RunContext.create(f"{run_id}-{suffix}-{child_timestamp}", RUNS_DIR)
    else:
        run_id = (
            run_id
            or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        )
        context = RunContext.create(str(run_id), RUNS_DIR, resume=resume)
        source = context.directory
    logger = configure_logging(context.directory, context.run_id)
    logger.info(
        "run start application=%s model=%s mode=%s",
        APPLICATION_VERSION,
        SCORING_MODEL_VERSION,
        "replay"
        if replay
        else "rescore"
        if rescore
        else "resume"
        if resume
        else "normal",
    )
    timings: dict[str, float] = {}
    store = None
    try:
        start = time.monotonic()
        if frozen_mode or context.checkpoint["universe_loaded"]:
            universe = _read(source / "universe.csv")
        else:
            universe = download_combined_constituents()
            atomic_csv(universe, context.directory / "universe.csv")
            context.checkpoint["universe_loaded"] = True
            context.save_checkpoint()
        timings["universe_load_seconds"] = time.monotonic() - start
        analysis_universe = universe[
            universe.get("mapping_status", pd.Series("VERIFIED", index=universe.index))
            .isin(["VERIFIED", "HEURISTIC"])
            & universe.symbol.notna()
        ].copy()

        start = time.monotonic()
        downloaded = None
        incremental_fetch_symbol_count = 0
        if frozen_mode or context.checkpoint["prices_complete"]:
            prices = _read(source / "price_features.csv")
            if not frozen_mode:
                benchmark_symbols = sorted({value[0] for value in BENCHMARKS.values()})
                resume_store = HistoryStore(HISTORY_DB)
                pending_symbols = resume_store.pending_outcome_symbols()
                resume_store.close()
                current_ids = dict(
                    zip(
                        analysis_universe.symbol,
                        analysis_universe.get("security_id", analysis_universe.symbol),
                    )
                )
                resume_symbols = list(
                    dict.fromkeys(
                        analysis_universe.symbol.tolist()
                        + sorted(pending_symbols)
                        + benchmark_symbols
                    )
                )
                price_db = sqlite3.connect(HISTORY_DB)
                local_prices = PriceStore(price_db)
                downloaded = local_prices.histories({
                    current_ids.get(symbol)
                    or local_prices.security_id_for_symbol(symbol)
                    or symbol: symbol
                    for symbol in resume_symbols
                })
                price_db.close()
        else:
            symbols = analysis_universe.symbol.tolist()
            benchmark_symbols = sorted({value[0] for value in BENCHMARKS.values()})
            pending_symbols = set()
            if HISTORY_DB.exists():
                pending_store = HistoryStore(HISTORY_DB)
                pending_symbols = pending_store.pending_outcome_symbols()
                pending_store.close()
            price_symbols = list(
                dict.fromkeys(symbols + sorted(pending_symbols) + benchmark_symbols)
            )
            price_db = sqlite3.connect(HISTORY_DB)
            local_prices = PriceStore(price_db)
            security_ids = dict(
                zip(
                    analysis_universe.symbol,
                    analysis_universe.get("security_id", analysis_universe.symbol),
                )
            )
            securities = {
                security_ids.get(symbol)
                or local_prices.security_id_for_symbol(symbol)
                or symbol: symbol
                for symbol in price_symbols
            }
            downloaded, incremental_fetch_symbol_count = _incremental_price_history(
                local_prices,
                securities,
                date.today() - timedelta(days=730),
                date.today(),
            )
            price_db.close()
            prices = build_price_features(downloaded, symbols)
            benchmark_prices = build_price_features(downloaded, benchmark_symbols)
            prices = attach_benchmark_prices(analysis_universe, prices, benchmark_prices)
            atomic_csv(prices, context.directory / "price_features.csv")
            context.checkpoint["prices_complete"] = True
            context.save_checkpoint()
        timings["price_download_seconds"] = time.monotonic() - start

        start = time.monotonic()
        provider = (
            _read(source / "normalized_provider.csv")
            if frozen_mode
            else _provider_data(context, analysis_universe, logger)
        )
        timings["provider_fetch_seconds"] = time.monotonic() - start

        if not frozen_mode:
            universe = _verify_provider_mappings(universe, provider)
            atomic_csv(universe, context.directory / "universe.csv")
            analysis_universe = universe[
                universe.get("mapping_status", pd.Series("VERIFIED", index=universe.index)).isin(["VERIFIED", "HEURISTIC"])
                & universe.symbol.notna()
            ].copy()
        constituent_sources = _constituent_health(universe, datetime.now(timezone.utc))
        universe_health_status = (
            "INVALID" if any(item["source_status"] == "INVALID" for item in constituent_sources.values())
            else "DEGRADED" if any(item["source_status"] == "DEGRADED" for item in constituent_sources.values())
            else "VALID"
        )
        if frozen_mode and not source_manifest.get("constituent_sources"):
            universe_health_status = "VALID"
        mapping_columns = ["source_index", "source_symbol", "company_name", "country",
                           "exchange", "isin", "mapping_status", "mapping_method", "mapping_error"]
        statuses = universe.get("mapping_status", pd.Series("VERIFIED", index=universe.index))
        atomic_csv(universe[~statuses.eq("VERIFIED")].reindex(columns=mapping_columns),
                   context.directory / "unmapped_constituents.csv")

        if not frozen_mode:
            context.freeze_analysis_as_of()
        elif source_manifest.get("analysis_as_of_utc"):
            context.checkpoint["analysis_as_of_utc"] = source_manifest[
                "analysis_as_of_utc"
            ]
            context.save_checkpoint()
        analysis_as_of = context.analysis_as_of or context.started

        store = HistoryStore(HISTORY_DB)
        if not frozen_mode:
            enriched = []
            for row in provider.to_dict("records"):
                row = store.add_analyst_history_features(
                    row["symbol"], row, analysis_as_of
                )
                for name in YahooClient.ANALYST_COMPONENTS:
                    if row.get(f"{name}_cache_status") == FRESH_PROVIDER and row.get(
                        f"{name}_fetched_at_utc"
                    ):
                        store.upsert_component(
                            row["symbol"],
                            name,
                            row,
                            datetime.fromisoformat(row[f"{name}_fetched_at_utc"]),
                        )
                enriched.append(row)
            provider = pd.DataFrame(enriched)
            atomic_csv(provider, context.directory / "normalized_provider.csv")

        start = time.monotonic()
        full, long_term, short_term = build_analysis(analysis_universe, prices, provider)
        replay_verification = {}
        if replay:
            source_full = _read(source / "full_analysis.csv").set_index("symbol")
            for column in (
                "long_term_score_change_7d",
                "short_term_score_change_7d",
                "long_term_rank_change_7d",
                "short_term_rank_change_7d",
                "rank_history_status",
            ):
                if column in source_full:
                    full[column] = full.symbol.map(source_full[column])
            replay_verification = verify_replay_output(source_full.reset_index(), full)
            atomic_json(
                context.directory / "replay_verification.json", replay_verification
            )
            if replay_verification["mismatch_count"]:
                raise RuntimeError(
                    "REPLAY_MISMATCH: recomputed outputs differ from frozen outputs"
                )
        else:
            full = _rank_changes(full, store, analysis_as_of)
        long_term = full[full.long_term_rank.notna()].sort_values("long_term_rank")
        short_term = full[full.short_term_rank.notna()].sort_values("short_term_rank")
        timings["scoring_seconds"] = time.monotonic() - start
        context.checkpoint["analysis_complete"] = True
        context.save_checkpoint()

        price_status = (
            prices["price_data_status"]
            if "price_data_status" in prices
            else pd.Series(np.where(prices.get("current_price", pd.Series(np.nan, index=prices.index)).notna(), "FRESH", "INSUFFICIENT"), index=prices.index)
        )
        metrics = {
            "universe_count": len(universe),
            "sp500_count": int(
                universe.index_name.str.contains("S&P 500", regex=False).sum()
            ),
            "stoxx600_count": int(
                universe.index_name.str.contains("STOXX Europe 600", regex=False).sum()
            ),
            "duplicate_membership_count": int(
                universe.index_name.str.contains("|", regex=False).sum()
            ),
            "price_present_coverage_pct": _coverage(prices, "current_price"),
            # Compatibility alias: explicitly means presence, never freshness.
            "price_coverage_pct": _coverage(prices, "current_price"),
            "price_fresh_coverage_pct": float(price_status.eq("FRESH").mean() * 100) if len(prices) else 0.0,
            "price_stale_coverage_pct": float(price_status.eq("STALE").mean() * 100) if len(prices) else 0.0,
            "price_insufficient_coverage_pct": float(price_status.isin(["INSUFFICIENT", "ERROR"]).mean() * 100) if len(prices) else 0.0,
            "analyst_coverage_pct": _coverage(full, "expectations_long_score"),
            "quality_coverage_pct": _coverage(full, "quality_score"),
            "growth_coverage_pct": _coverage(full, "growth_score"),
            "valuation_coverage_pct": _coverage(full, "valuation_score"),
            "expectations_coverage_pct": _coverage(full, "expectations_long_score"),
            "long_expectations_coverage_pct": _coverage(
                full, "expectations_long_score"
            ),
            "short_expectations_coverage_pct": _coverage(
                full, "expectations_short_score"
            ),
            "short_rs_coverage_pct": _coverage(full, "short_rs_score"),
            "setup_data_coverage_pct": _coverage(full, "setup_quality_score"),
            "technical_data_coverage_pct": _coverage(full, "technical_trend_score"),
            "lt_score_coverage_pct": _coverage(full, "long_term_score"),
            "lt_ranked_count": len(long_term),
            "st_ranked_count": len(short_term),
            "price_missing_symbols_count": int(prices.current_price.isna().sum()),
            "provider_error_symbol_count": int(
                provider.get("data_errors", pd.Series("", index=provider.index))
                .fillna("")
                .ne("")
                .sum()
            ),
            "constituent_sources": constituent_sources,
            "fcf_yield_available_pct": _coverage(full, "fcf_yield"),
            "fcf_currency_mismatch_count": int(
                full.get("fcf_yield_currency_status", pd.Series(dtype=str))
                .eq("CURRENCY_MISMATCH")
                .sum()
            ),
            "fcf_currency_unavailable_count": int(
                full.get("fcf_yield_currency_status", pd.Series(dtype=str))
                .eq("CURRENCY_UNAVAILABLE")
                .sum()
            ),
            "incremental_price_fetch_symbol_count": incremental_fetch_symbol_count,
        }
        statuses = provider.filter(regex="cache_status$").astype(str)
        for name, value in (
            ("fresh_provider_count", FRESH_PROVIDER),
            ("fresh_cache_count", FRESH_CACHE),
            ("stale_fallback_count", STALE_FALLBACK),
            ("partial_count", PARTIAL),
            ("error_count", ERROR),
        ):
            metrics[name] = int(statuses.eq(value).sum().sum())
        for component in (
            *YahooClient.ANALYST_COMPONENTS,
            "earnings_dates",
            "info",
            "valuation",
            "fundamental",
        ):
            component_status = provider.get(
                f"{component}_cache_status", pd.Series("", index=provider.index)
            ).astype(str)
            metrics[f"provider_contract_{component}"] = {
                "fresh_provider_count": int(component_status.eq(FRESH_PROVIDER).sum()),
                "fresh_cache_count": int(component_status.eq(FRESH_CACHE).sum()),
                "error_count": int(component_status.eq(ERROR).sum()),
                "stale_fallback_count": int(component_status.eq(STALE_FALLBACK).sum()),
                "usable_count": int(component_status.ne(ERROR).sum()),
            }
        for prefix, column in (
            ("lt_score", "long_term_score"),
            ("st_score", "short_term_score"),
        ):
            numeric = pd.to_numeric(full.get(column), errors="coerce").dropna()
            metrics[f"{prefix}_median"] = (
                float(numeric.median()) if len(numeric) else None
            )
            metrics[f"{prefix}_p10"] = (
                float(numeric.quantile(0.1)) if len(numeric) else None
            )
            metrics[f"{prefix}_p90"] = (
                float(numeric.quantile(0.9)) if len(numeric) else None
            )
        metrics["risk_median"] = (
            float(pd.to_numeric(full.get("risk_score"), errors="coerce").median())
            if "risk_score" in full
            else None
        )
        metrics["confidence_median"] = (
            float(pd.to_numeric(full.get("confidence_score"), errors="coerce").median())
            if "confidence_score" in full
            else None
        )
        total_statuses = int(statuses.size)
        metrics["cache_hit_ratio"] = (
            metrics["fresh_cache_count"] / total_statuses if total_statuses else 0
        )
        metrics["fresh_fetch_ratio"] = (
            metrics["fresh_provider_count"] / total_statuses if total_statuses else 0
        )
        health_details = horizon_run_statuses(metrics, universe_valid=universe_health_status)
        universe_reasons = [
            f"{key}: {reason}"
            for key, item in constituent_sources.items()
            for reason in item["status_reasons"]
        ]
        health_details["common_run_status_reasons"] = (
            universe_reasons + health_details["common_run_status_reasons"]
        )
        health = health_details["overall_run_status"]
        reasons = (
            health_details["common_run_status_reasons"]
            + health_details["lt_run_status_reasons"]
            + health_details["st_run_status_reasons"]
        )
        metrics.update(
            {
                "run_status": health,
                "run_status_reasons": reasons,
                **health_details,
                "benchmark_rs_status": "AVAILABLE_EXPERIMENTAL"
                if full.get("benchmark_rs_status", pd.Series("", index=full.index))
                .eq("AVAILABLE_EXPERIMENTAL")
                .any()
                else "FALLBACK_RAW",
                **{key: round(value, 3) for key, value in timings.items()},
            }
        )
        for benchmark_symbol in ("^GSPC", "^STOXX"):
            members = full.get("benchmark_symbol", pd.Series("", index=full.index)).eq(
                benchmark_symbol
            )
            available = (
                full.loc[members]
                .get("benchmark_price", pd.Series(dtype=float))
                .notna()
                .any()
                if members.any()
                else False
            )
            metrics[f"benchmark_{benchmark_symbol}_status"] = (
                "AVAILABLE_EXPERIMENTAL" if available else "UNAVAILABLE"
            )
            observed = (
                full.loc[members]
                .get("benchmark_price_as_of", pd.Series(dtype=object))
                .dropna()
            )
            metrics[f"benchmark_{benchmark_symbol}_price_as_of"] = (
                observed.max() if len(observed) else None
            )

        for row in prices[prices.current_price.isna()].to_dict("records"):
            context.record_error(
                row["symbol"],
                "PRICE",
                "OHLCV",
                "price unavailable after retries",
                error_type="MISSING_PRICE_DATA",
            )
        for row in provider[
            provider.get("data_errors", pd.Series("", index=provider.index))
            .fillna("")
            .ne("")
        ].to_dict("records"):
            context.record_error(
                row["symbol"], "ANALYST", "yahoo", row.get("data_errors")
            )
        for row in provider.to_dict("records"):
            for component in (
                *YahooClient.ANALYST_COMPONENTS,
                "earnings_dates",
                "info",
                "valuation",
                "fundamental",
            ):
                if row.get(f"{component}_cache_status") == ERROR:
                    stage = (
                        "ANALYST"
                        if component in YahooClient.ANALYST_COMPONENTS
                        else component.upper()
                    )
                    context.record_error(
                        row["symbol"],
                        stage,
                        component,
                        row.get(f"{component}_error_message")
                        or "provider component error",
                    )
                elif row.get(f"{component}_cache_status") == STALE_FALLBACK:
                    stage = (
                        "ANALYST"
                        if component in YahooClient.ANALYST_COMPONENTS
                        else component.upper()
                    )
                    context.record_error(
                        row["symbol"],
                        stage,
                        component,
                        "stale cache fallback used",
                        stale=True,
                        error_type="STALE_FALLBACK",
                    )
        context.save_errors()
        start = time.monotonic()
        export_run(
            context.directory,
            full,
            long_term,
            short_term,
            metrics,
            methodology(),
            health_details["lt_run_status"],
            health_details["st_run_status"],
        )
        timings["export_seconds"] = time.monotonic() - start
        metrics["export_seconds"] = round(timings["export_seconds"], 3)
        outcomes_updated = 0
        outcome_start = time.monotonic()
        if not frozen_mode and not context.checkpoint["history_saved"]:
            ranked = _persistence_rows(full, health_details)
            if ranked:
                store.save_rankings(context.run_id, analysis_as_of.isoformat(), ranked)
                store.save_predictions(
                    context.run_id,
                    analysis_as_of.isoformat(),
                    SCORING_MODEL_VERSION,
                    ranked,
                    health_details["lt_run_status"],
                    health_details["st_run_status"],
                    health_details["overall_run_status"],
                )
            context.checkpoint["history_saved"] = True
            context.save_checkpoint()
        if not frozen_mode and downloaded is not None:
            from investment_ai.validation import update_outcomes, validation_report

            symbols = analysis_universe.symbol.tolist()
            benchmark_symbols = sorted({value[0] for value in BENCHMARKS.values()})
            outcome_symbols = sorted(set(symbols) | store.pending_outcome_symbols())
            histories = {
                symbol: symbol_history(downloaded, symbol).get("Close")
                for symbol in outcome_symbols + benchmark_symbols
            }
            outcomes_updated = update_outcomes(
                store.db,
                {s: histories[s] for s in outcome_symbols if histories[s] is not None},
                {
                    s: histories[s]
                    for s in benchmark_symbols
                    if histories[s] is not None
                },
            )
            context.checkpoint["outcomes_updated"] = True
            from investment_ai.reporting.export import write_validation_snapshot

            write_validation_snapshot(
                context.directory / "validation_snapshot.json",
                validation_report(store.db),
            )
        timings["outcome_update_seconds"] = time.monotonic() - outcome_start
        metrics["outcome_update_seconds"] = round(timings["outcome_update_seconds"], 3)
        metrics["outcomes_updated_count"] = outcomes_updated
        context.checkpoint["exports_complete"] = True
        context.save_checkpoint()
        config = {
            "force_refresh": FORCE_REFRESH,
            "export_results": EXPORT_RESULTS,
            "max_workers": MAX_WORKERS,
            "price_batch_size": PRICE_BATCH_SIZE,
            "minimum_peers": MIN_PEERS,
            "top_n": TOP_N,
            "price_period": PRICE_PERIOD,
        }
        metrics.update(
            {
                "force_refresh": FORCE_REFRESH,
                "export_results": EXPORT_RESULTS,
                "max_workers": MAX_WORKERS,
                "price_batch_size": PRICE_BATCH_SIZE,
                "minimum_peers": MIN_PEERS,
                "replay_mode": "EXACT_COMPATIBLE"
                if replay
                else "RESCORE"
                if rescore
                else "NORMAL",
                "mode": "replay" if replay else "rescore" if rescore else "normal",
                "source_run_id": run_id if frozen_mode else None,
                "source_application_version": source_manifest.get("application_version")
                if frozen_mode
                else None,
                "source_scoring_model_version": source_manifest.get(
                    "scoring_model_version"
                )
                if frozen_mode
                else None,
                **replay_verification,
                "total_duration_seconds": round(
                    (datetime.now(timezone.utc) - context.started).total_seconds(), 3
                ),
            }
        )
        build_manifest(context, metrics, config)
        logger.info("run health status=%s reasons=%s", health, reasons)
        if (
            health_details["lt_run_status"] == "INVALID"
            and health_details["st_run_status"] == "INVALID"
        ):
            print(
                "\nINVALID RUN — rankings suppressed; see run_manifest.json and errors.csv"
            )
            logger.error("run end invalid")
            return 3
        if health == "DEGRADED":
            print("\n*** DEGRADED RUN — review coverage warnings before use ***")
        print_rankings(
            long_term
            if health_details["lt_run_status"] != "INVALID"
            else long_term.iloc[0:0],
            short_term
            if health_details["st_run_status"] != "INVALID"
            else short_term.iloc[0:0],
            TOP_N,
        )
        print(f"""\nApplication: {APPLICATION_VERSION}
Scoring model: {SCORING_MODEL_VERSION}
Run ID: {context.run_id}
Analysis as-of: {context.checkpoint.get("analysis_as_of_utc")}
Common status: {health_details["common_run_status"]}
LT status: {health_details["lt_run_status"]}
ST status: {health_details["st_run_status"]}
Universe: {metrics["universe_count"]} | LT ranked: {metrics["lt_ranked_count"]} | ST ranked: {metrics["st_ranked_count"]}
Price present/fresh coverage: {metrics["price_present_coverage_pct"]:.1f}%/{metrics["price_fresh_coverage_pct"]:.1f}% | Quality coverage: {metrics["quality_coverage_pct"]:.1f}% | Growth coverage: {metrics["growth_coverage_pct"]:.1f}%
Valuation coverage: {metrics["valuation_coverage_pct"]:.1f}% | Long Expectations: {metrics["long_expectations_coverage_pct"]:.1f}% | Short Expectations: {metrics["short_expectations_coverage_pct"]:.1f}%
Provider errors: {metrics["provider_error_symbol_count"]} | Stale fallbacks: {metrics["stale_fallback_count"]} | Outcome rows updated: {outcomes_updated}
Constituents: S&P {constituent_sources['sp500']['source_status']} ({constituent_sources['sp500']['mapping_success_pct']:.1f}% usable) | STOXX {constituent_sources['stoxx600']['source_status']} ({constituent_sources['stoxx600']['mapping_success_pct']:.1f}% usable)
Artifacts: {context.directory}""")
        logger.info("run end")
        return 0
    except Exception as exc:
        context.record_error("", "SCORING", "run", exc)
        context.save_errors()
        build_manifest(
            context,
            {
                "run_status": "INVALID",
                "run_status_reasons": [str(exc)],
                "universe_count": 0,
                "sp500_count": 0,
                "stoxx600_count": 0,
                "duplicate_membership_count": 0,
                "price_coverage_pct": 0,
                "analyst_coverage_pct": 0,
                "quality_coverage_pct": 0,
                "growth_coverage_pct": 0,
                "valuation_coverage_pct": 0,
                "expectations_coverage_pct": 0,
                "lt_ranked_count": 0,
                "st_ranked_count": 0,
                "fresh_provider_count": 0,
                "fresh_cache_count": 0,
                "stale_fallback_count": 0,
                "partial_count": 0,
                "error_count": 1,
                "price_missing_symbols_count": 0,
                "provider_error_symbol_count": 1,
                "force_refresh": FORCE_REFRESH,
                "export_results": EXPORT_RESULTS,
                "max_workers": MAX_WORKERS,
                "price_batch_size": PRICE_BATCH_SIZE,
                "minimum_peers": MIN_PEERS,
            },
            {"fatal": True},
        )
        logger.exception("fatal run exception")
        raise
    finally:
        if store is not None:
            store.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args([] if argv is None else argv)
    if args.validation_report:
        from investment_ai.validation import validation_report

        store = HistoryStore(HISTORY_DB)
        print(json.dumps(validation_report(store.db), indent=2, default=str))
        store.close()
        return 0
    return execute(
        args.replay or args.rescore or args.resume,
        resume=bool(args.resume),
        replay=bool(args.replay),
        rescore=bool(args.rescore),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ValueError as exc:
        print(exc)
        raise SystemExit(2)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(4 if "schema" in str(exc).lower() else 2)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        raise SystemExit(1)

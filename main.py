"""Investment AI v1.0 entry point. Canonical usage: ``python main.py``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from investment_ai.config import (
    APPLICATION_VERSION, CACHE_DIR, EXPORT_RESULTS, FORCE_REFRESH, HISTORY_DB,
    MAX_WORKERS, MIN_PEERS, PRICE_BATCH_SIZE, PRICE_PERIOD, RUNS_DIR,
    SCORING_MODEL_VERSION, TOP_N, validate_config,
)
from investment_ai.data.cache import JsonCache
from investment_ai.data.constituents import fetch_index_constituents
from investment_ai.data.history_store import HistoryStore
from investment_ai.data.prices import build_price_features, download_prices, symbol_history
from investment_ai.data.yahoo import YahooClient
from investment_ai.features.benchmark import BENCHMARKS, attach_benchmark_prices
from investment_ai.pipeline import build_analysis
from investment_ai.reporting.export import export_run
from investment_ai.reporting.tables import print_rankings
from investment_ai.runtime import (
    RunContext, atomic_csv, build_manifest, configure_logging, horizon_run_statuses,
)
from investment_ai.status import ERROR, FRESH_CACHE, FRESH_PROVIDER, PARTIAL, STALE_FALLBACK


def download_combined_constituents() -> pd.DataFrame:
    frame = fetch_index_constituents(CACHE_DIR / "constituents")
    present = {item.strip() for value in frame.index_name.dropna() for item in str(value).split("|")}
    if not {"S&P 500", "STOXX Europe 600"} <= present:
        raise RuntimeError("Both constituent sources are required for the combined universe")
    return frame


def methodology() -> pd.DataFrame:
    return pd.DataFrame([
        {"score": "Long term", "formula": "25% Quality + 20% Growth + 20% Valuation + 20% Long-Term Expectations + 10% Long Trend + 5% Financial Safety"},
        {"score": "Short term", "formula": "20% raw Relative Strength + 25% Setup + 20% Short-Term Expectations + 10% Volume + 15% Technical Trend + 10% Event Timing"},
        {"score": "Benchmark RS", "formula": "stock return - assigned regional benchmark return; experimental and retained beside raw RS"},
    ])


def _rank_changes(frame: pd.DataFrame, store: HistoryStore, started: datetime) -> pd.DataFrame:
    result = frame.copy()
    values = []
    for row in result.to_dict("records"):
        old = store.changes(row["symbol"], 7, started)
        if not old:
            values.append((np.nan, np.nan, np.nan, np.nan, "NEW"))
        else:
            changes = []
            for key, rank in (("long_term_score", False), ("short_term_score", False),
                              ("long_term_rank", True), ("short_term_rank", True)):
                current, previous = row.get(key), old[key]
                changes.append(
                    (previous - current if rank else current - previous)
                    if pd.notna(current) and previous is not None else np.nan
                )
            values.append(tuple(changes) + ("AVAILABLE",))
    result[["long_term_score_change_7d", "short_term_score_change_7d",
            "long_term_rank_change_7d", "short_term_rank_change_7d",
            "rank_history_status"]] = values
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic investment research ranking")
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
    artifact_symbols = set(prior.symbol.astype(str)) if len(prior) and "symbol" in prior else set()
    # The artifact is the durable truth; a checkpoint cannot claim an absent row.
    complete = set(context.checkpoint["provider_symbols_complete"]) & artifact_symbols
    context.checkpoint["provider_symbols_complete"] = sorted(complete)
    # Failed symbols are intentionally absent from complete and retried on resume.
    pending = universe[~universe.symbol.isin(complete)]
    for offset in range(0, len(pending), 25):
        batch = pending.iloc[offset:offset + 25]
        logger.info("provider fetch start symbols=%d", len(pending))
        fresh = YahooClient(JsonCache(CACHE_DIR)).fetch_many(batch)
        prior = pd.concat([prior[~prior.symbol.isin(fresh.symbol)] if len(prior) else prior, fresh], ignore_index=True)
        success = pd.to_numeric(
            fresh.get("provider_success", pd.Series(1, index=fresh.index)), errors="coerce"
        ).gt(0)
        complete_now = fresh.loc[success, "symbol"].tolist()
        failed_now = fresh.loc[~fresh.symbol.isin(complete_now), "symbol"].tolist()
        context.checkpoint["provider_symbols_complete"] = sorted(complete | set(complete_now))
        context.checkpoint["provider_symbols_failed"] = sorted(failed_now)
        atomic_csv(prior, path)
        context.save_checkpoint()
        logger.info("provider fetch end complete=%d failed=%d", len(complete_now), len(failed_now))
        complete.update(complete_now)
    context.checkpoint["provider_complete"] = set(universe.symbol) <= set(prior.get("symbol", []))
    context.save_checkpoint()
    return prior


def _coverage(frame: pd.DataFrame, column: str) -> float:
    return float(frame.get(column, pd.Series(np.nan, index=frame.index)).notna().mean() * 100) if len(frame) else 0.0


def _verify_source(source: Path, exact: bool) -> dict:
    required = ("run_manifest.json", "universe.csv", "price_features.csv",
                "normalized_provider.csv", "full_analysis.csv")
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise RuntimeError(f"Source run is missing required artifacts: {', '.join(missing)}")
    manifest = json.loads((source / "run_manifest.json").read_text())
    if exact and manifest.get("scoring_model_version") != SCORING_MODEL_VERSION:
        raise RuntimeError("Exact replay requires the source scoring model version")
    if exact:
        for name, digest in manifest.get("artifact_sha256", {}).items():
            actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
            if actual != digest:
                raise RuntimeError(f"Exact replay artifact checksum mismatch: {name}")
    return manifest


def execute(run_id: str | None = None, resume: bool = False, replay: bool = False,
            rescore: bool = False) -> int:
    validate_config()
    frozen_mode = replay or rescore
    source_manifest = None
    if frozen_mode:
        source = RUNS_DIR / str(run_id)
        if not source.exists():
            raise FileNotFoundError(f"Replay run not found: {run_id}")
        source_manifest = _verify_source(source, exact=replay)
        suffix = "replay" if replay else "rescore"
        context = RunContext.create(f"{run_id}-{suffix}", RUNS_DIR)
    else:
        run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        context = RunContext.create(str(run_id), RUNS_DIR, resume=resume)
        source = context.directory
    logger = configure_logging(context.directory, context.run_id)
    logger.info("run start application=%s model=%s mode=%s", APPLICATION_VERSION, SCORING_MODEL_VERSION,
                "replay" if replay else "rescore" if rescore else "resume" if resume else "normal")
    timings: dict[str, float] = {}
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

        start = time.monotonic()
        downloaded = None
        if frozen_mode or context.checkpoint["prices_complete"]:
            prices = _read(source / "price_features.csv")
        else:
            symbols = universe.symbol.tolist()
            benchmark_symbols = sorted({value[0] for value in BENCHMARKS.values()})
            downloaded = download_prices(symbols + benchmark_symbols, PRICE_PERIOD)
            prices = build_price_features(downloaded, symbols)
            benchmark_prices = build_price_features(downloaded, benchmark_symbols)
            prices = attach_benchmark_prices(universe, prices, benchmark_prices)
            atomic_csv(prices, context.directory / "price_features.csv")
            context.checkpoint["prices_complete"] = True
            context.save_checkpoint()
        timings["price_download_seconds"] = time.monotonic() - start

        start = time.monotonic()
        provider = _read(source / "normalized_provider.csv") if frozen_mode else _provider_data(context, universe, logger)
        timings["provider_fetch_seconds"] = time.monotonic() - start

        if not frozen_mode:
            context.freeze_analysis_as_of()
        elif source_manifest.get("analysis_as_of_utc"):
            context.checkpoint["analysis_as_of_utc"] = source_manifest["analysis_as_of_utc"]
            context.save_checkpoint()
        analysis_as_of = context.analysis_as_of or context.started

        store = HistoryStore(HISTORY_DB)
        if not frozen_mode:
            enriched = []
            for row in provider.to_dict("records"):
                row = store.add_analyst_history_features(row["symbol"], row, analysis_as_of)
                for name in YahooClient.ANALYST_COMPONENTS:
                    if row.get(f"{name}_cache_status") == FRESH_PROVIDER and row.get(f"{name}_fetched_at_utc"):
                        store.upsert_component(row["symbol"], name, row, datetime.fromisoformat(row[f"{name}_fetched_at_utc"]))
                enriched.append(row)
            provider = pd.DataFrame(enriched)
            atomic_csv(provider, context.directory / "normalized_provider.csv")

        start = time.monotonic()
        full, long_term, short_term = build_analysis(universe, prices, provider)
        if replay:
            source_full = _read(source / "full_analysis.csv").set_index("symbol")
            for column in ("long_term_score_change_7d", "short_term_score_change_7d",
                           "long_term_rank_change_7d", "short_term_rank_change_7d", "rank_history_status"):
                if column in source_full:
                    full[column] = full.symbol.map(source_full[column])
        else:
            full = _rank_changes(full, store, analysis_as_of)
        long_term = full[full.long_term_rank.notna()].sort_values("long_term_rank")
        short_term = full[full.short_term_rank.notna()].sort_values("short_term_rank")
        timings["scoring_seconds"] = time.monotonic() - start
        context.checkpoint["analysis_complete"] = True
        context.save_checkpoint()

        metrics = {
            "universe_count": len(universe),
            "sp500_count": int(universe.index_name.str.contains("S&P 500", regex=False).sum()),
            "stoxx600_count": int(universe.index_name.str.contains("STOXX Europe 600", regex=False).sum()),
            "duplicate_membership_count": int(universe.index_name.str.contains("|", regex=False).sum()),
            "price_coverage_pct": _coverage(prices, "current_price"),
            "analyst_coverage_pct": _coverage(full, "expectations_long_score"),
            "quality_coverage_pct": _coverage(full, "quality_score"),
            "growth_coverage_pct": _coverage(full, "growth_score"),
            "valuation_coverage_pct": _coverage(full, "valuation_score"),
            "expectations_coverage_pct": _coverage(full, "expectations_long_score"),
            "long_expectations_coverage_pct": _coverage(full, "expectations_long_score"),
            "short_expectations_coverage_pct": _coverage(full, "expectations_short_score"),
            "short_rs_coverage_pct": _coverage(full, "short_rs_score"),
            "setup_data_coverage_pct": _coverage(full, "setup_quality_score"),
            "technical_data_coverage_pct": _coverage(full, "technical_trend_score"),
            "lt_score_coverage_pct": _coverage(full, "long_term_score"),
            "lt_ranked_count": len(long_term), "st_ranked_count": len(short_term),
            "price_missing_symbols_count": int(prices.current_price.isna().sum()),
            "provider_error_symbol_count": int(provider.get("data_errors", pd.Series("", index=provider.index)).fillna("").ne("").sum()),
        }
        statuses = provider.filter(regex="cache_status$").astype(str)
        for name, value in (("fresh_provider_count", FRESH_PROVIDER), ("fresh_cache_count", FRESH_CACHE),
                            ("stale_fallback_count", STALE_FALLBACK), ("partial_count", PARTIAL),
                            ("error_count", ERROR)):
            metrics[name] = int(statuses.eq(value).sum().sum())
        for component in YahooClient.ANALYST_COMPONENTS:
            component_status = provider.get(
                f"{component}_cache_status", pd.Series("", index=provider.index)
            ).astype(str)
            metrics[f"provider_contract_{component}"] = {
                "success_count": int(component_status.isin([FRESH_PROVIDER, FRESH_CACHE]).sum()),
                "parse_empty_count": int(provider.get(
                    f"{component}_success", pd.Series(False, index=provider.index)
                ).eq(False).sum()),
                "error_count": int(component_status.eq(ERROR).sum()),
                "stale_fallback_count": int(component_status.eq(STALE_FALLBACK).sum()),
            }
        for prefix, column in (("lt_score", "long_term_score"), ("st_score", "short_term_score")):
            numeric = pd.to_numeric(full.get(column), errors="coerce").dropna()
            metrics[f"{prefix}_median"] = float(numeric.median()) if len(numeric) else None
            metrics[f"{prefix}_p10"] = float(numeric.quantile(.1)) if len(numeric) else None
            metrics[f"{prefix}_p90"] = float(numeric.quantile(.9)) if len(numeric) else None
        metrics["risk_median"] = float(pd.to_numeric(full.get("risk_score"), errors="coerce").median()) if "risk_score" in full else None
        metrics["confidence_median"] = float(pd.to_numeric(full.get("confidence_score"), errors="coerce").median()) if "confidence_score" in full else None
        total_statuses = int(statuses.size)
        metrics["cache_hit_ratio"] = metrics["fresh_cache_count"] / total_statuses if total_statuses else 0
        metrics["fresh_fetch_ratio"] = metrics["fresh_provider_count"] / total_statuses if total_statuses else 0
        health_details = horizon_run_statuses(metrics)
        health = health_details["overall_run_status"]
        reasons = (health_details["common_run_status_reasons"] +
                   health_details["lt_run_status_reasons"] + health_details["st_run_status_reasons"])
        metrics.update({"run_status": health, "run_status_reasons": reasons, **health_details,
                        "benchmark_rs_status": "AVAILABLE_EXPERIMENTAL" if full.get(
                            "benchmark_rs_status", pd.Series("", index=full.index)
                        ).eq("AVAILABLE_EXPERIMENTAL").any() else "FALLBACK_RAW",
                        **{key: round(value, 3) for key, value in timings.items()}})
        for benchmark_symbol in ("^GSPC", "^STOXX"):
            members = full.get("benchmark_symbol", pd.Series("", index=full.index)).eq(benchmark_symbol)
            available = full.loc[members].get(
                "benchmark_price", pd.Series(dtype=float)
            ).notna().any() if members.any() else False
            metrics[f"benchmark_{benchmark_symbol}_status"] = (
                "AVAILABLE_EXPERIMENTAL" if available else "UNAVAILABLE"
            )
            observed = full.loc[members].get("benchmark_price_as_of", pd.Series(dtype=object)).dropna()
            metrics[f"benchmark_{benchmark_symbol}_price_as_of"] = observed.max() if len(observed) else None

        for row in prices[prices.current_price.isna()].to_dict("records"):
            context.record_error(row["symbol"], "PRICE", "OHLCV", "price unavailable after retries",
                                 error_type="MISSING_PRICE_DATA")
        for row in provider[provider.get("data_errors", pd.Series("", index=provider.index)).fillna("").ne("")].to_dict("records"):
            context.record_error(row["symbol"], "ANALYST", "yahoo", row.get("data_errors"))
        for row in provider.to_dict("records"):
            for component in (*YahooClient.ANALYST_COMPONENTS, "valuation", "fundamental"):
                if row.get(f"{component}_cache_status") == ERROR:
                    stage = "ANALYST" if component in YahooClient.ANALYST_COMPONENTS else component.upper()
                    context.record_error(row["symbol"], stage, component,
                                         row.get(f"{component}_error_message") or "provider component error")
                elif row.get(f"{component}_cache_status") == STALE_FALLBACK:
                    stage = "ANALYST" if component in YahooClient.ANALYST_COMPONENTS else component.upper()
                    context.record_error(row["symbol"], stage, component, "stale cache fallback used",
                                         stale=True, error_type="STALE_FALLBACK")
        context.save_errors()
        start = time.monotonic()
        export_run(context.directory, full, long_term, short_term, metrics, methodology(),
                   health_details["lt_run_status"], health_details["st_run_status"])
        timings["export_seconds"] = time.monotonic() - start
        metrics["export_seconds"] = round(timings["export_seconds"], 3)
        outcomes_updated = 0
        outcome_start = time.monotonic()
        if not frozen_mode and not context.checkpoint["history_saved"]:
            ranked = full[full.long_term_rank.notna() | full.short_term_rank.notna()].to_dict("records")
            store.save_rankings(context.run_id, analysis_as_of.isoformat(), ranked)
            store.save_predictions(context.run_id, analysis_as_of.isoformat(), SCORING_MODEL_VERSION, ranked)
            context.checkpoint["history_saved"] = True
        if not frozen_mode and downloaded is not None:
            from investment_ai.validation import update_outcomes
            symbols = universe.symbol.tolist()
            benchmark_symbols = sorted({value[0] for value in BENCHMARKS.values()})
            histories = {symbol: symbol_history(downloaded, symbol).get("Close")
                         for symbol in symbols + benchmark_symbols}
            outcomes_updated = update_outcomes(
                store.db, {s: histories[s] for s in symbols if histories[s] is not None},
                {s: histories[s] for s in benchmark_symbols if histories[s] is not None})
            context.checkpoint["outcomes_updated"] = True
        timings["outcome_update_seconds"] = time.monotonic() - outcome_start
        metrics["outcome_update_seconds"] = round(timings["outcome_update_seconds"], 3)
        metrics["outcomes_updated_count"] = outcomes_updated
        context.checkpoint["exports_complete"] = True
        context.save_checkpoint()
        store.close()
        config = {"force_refresh": FORCE_REFRESH, "export_results": EXPORT_RESULTS,
                  "max_workers": MAX_WORKERS, "price_batch_size": PRICE_BATCH_SIZE,
                  "minimum_peers": MIN_PEERS, "top_n": TOP_N, "price_period": PRICE_PERIOD}
        metrics.update({"force_refresh": FORCE_REFRESH, "export_results": EXPORT_RESULTS,
                        "max_workers": MAX_WORKERS, "price_batch_size": PRICE_BATCH_SIZE,
                        "minimum_peers": MIN_PEERS,
                        "replay_mode": "EXACT_COMPATIBLE" if replay else "RESCORE" if rescore else "NORMAL",
                        "source_run_id": run_id if frozen_mode else None,
                        "source_application_version": source_manifest.get("application_version") if frozen_mode else None,
                        "source_scoring_model_version": source_manifest.get("scoring_model_version") if frozen_mode else None,
                        "total_duration_seconds": round((datetime.now(timezone.utc) - context.started).total_seconds(), 3)})
        manifest = build_manifest(context, metrics, config)
        logger.info("run health status=%s reasons=%s", health, reasons)
        if health_details["lt_run_status"] == "INVALID" and health_details["st_run_status"] == "INVALID":
            print("\nINVALID RUN — rankings suppressed; see run_manifest.json and errors.csv")
            logger.error("run end invalid")
            return 3
        if health == "DEGRADED":
            print("\n*** DEGRADED RUN — review coverage warnings before use ***")
        print_rankings(long_term if health_details["lt_run_status"] != "INVALID" else long_term.iloc[0:0],
                       short_term if health_details["st_run_status"] != "INVALID" else short_term.iloc[0:0], TOP_N)
        print(f"\nArtifacts: {context.directory}\nRun status: {manifest['run_status']}")
        logger.info("run end")
        return 0
    except Exception as exc:
        context.record_error("", "SCORING", "run", exc)
        context.save_errors()
        build_manifest(context, {
            "run_status": "INVALID", "run_status_reasons": [str(exc)],
            "universe_count": 0, "sp500_count": 0, "stoxx600_count": 0,
            "duplicate_membership_count": 0, "price_coverage_pct": 0,
            "analyst_coverage_pct": 0, "quality_coverage_pct": 0,
            "growth_coverage_pct": 0, "valuation_coverage_pct": 0,
            "expectations_coverage_pct": 0, "lt_ranked_count": 0,
            "st_ranked_count": 0, "fresh_provider_count": 0,
            "fresh_cache_count": 0, "stale_fallback_count": 0,
            "partial_count": 0, "error_count": 1,
            "price_missing_symbols_count": 0, "provider_error_symbol_count": 1,
            "force_refresh": FORCE_REFRESH, "export_results": EXPORT_RESULTS,
            "max_workers": MAX_WORKERS, "price_batch_size": PRICE_BATCH_SIZE,
            "minimum_peers": MIN_PEERS,
        }, {"fatal": True})
        logger.exception("fatal run exception")
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args([] if argv is None else argv)
    if args.validation_report:
        from investment_ai.validation import validation_report
        store = HistoryStore(HISTORY_DB)
        print(json.dumps(validation_report(store.db), indent=2, default=str))
        store.close()
        return 0
    return execute(args.replay or args.rescore or args.resume, resume=bool(args.resume),
                   replay=bool(args.replay), rescore=bool(args.rescore))


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

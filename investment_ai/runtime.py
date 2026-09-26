"""Run lifecycle, durable artifacts, health gates, and operational telemetry."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from investment_ai.config import (
    APPLICATION_VERSION, CACHE_SCHEMA_VERSION, DATABASE_SCHEMA_VERSION,
    SCORING_MODEL_VERSION,
)

ERROR_COLUMNS = [
    "symbol", "stage", "component", "error_type", "error_message",
    "retry_count", "used_stale_fallback", "timestamp_utc", "attempt_number",
]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_status(metrics: dict[str, Any], universe_valid: bool = True) -> tuple[str, list[str]]:
    required = {
        "price_coverage_pct": (97, 90),
        "quality_coverage_pct": (70, 55),
        "valuation_coverage_pct": (70, 55),
        "expectations_coverage_pct": (75, 60),
    }
    reasons = []
    if not universe_valid:
        return "INVALID", ["combined universe validation failed"]
    for key, (_, floor) in required.items():
        value = float(metrics.get(key, 0) or 0)
        if value < floor:
            reasons.append(f"{key}={value:.1f}% is below {floor}%")
    if reasons:
        return "INVALID", reasons
    degraded = []
    for key, (valid, _) in required.items():
        value = float(metrics.get(key, 0) or 0)
        if value < valid:
            degraded.append(f"{key}={value:.1f}% is below {valid}%")
    return ("DEGRADED", degraded) if degraded else ("VALID", [])


def horizon_run_statuses(metrics: dict[str, Any], universe_valid: bool = True) -> dict[str, Any]:
    """Evaluate shared, long-term and short-term input readiness independently."""
    def evaluate(spec: dict[str, tuple[float, float]]) -> tuple[str, list[str]]:
        invalid = [f"{k}={float(metrics.get(k, 0) or 0):.1f}% is below {floor}%"
                   for k, (_, floor) in spec.items() if float(metrics.get(k, 0) or 0) < floor]
        if invalid:
            return "INVALID", invalid
        degraded = [f"{k}={float(metrics.get(k, 0) or 0):.1f}% is below {target}%"
                    for k, (target, _) in spec.items() if float(metrics.get(k, 0) or 0) < target]
        return ("DEGRADED", degraded) if degraded else ("VALID", [])

    common = ("INVALID", ["combined universe validation failed"]) if not universe_valid else evaluate({
        "price_coverage_pct": (97, 90),
    })
    lt = evaluate({
        "quality_coverage_pct": (70, 55), "growth_coverage_pct": (70, 55),
        "valuation_coverage_pct": (70, 55), "long_expectations_coverage_pct": (75, 60),
        "lt_score_coverage_pct": (75, 60),
    })
    st = evaluate({
        "price_coverage_pct": (97, 90), "short_expectations_coverage_pct": (75, 60),
        "short_rs_coverage_pct": (75, 60), "setup_data_coverage_pct": (75, 60),
        "technical_data_coverage_pct": (75, 60),
    })
    if common[0] == "INVALID":
        lt = st = ("INVALID", common[1])
    order = {"VALID": 0, "DEGRADED": 1, "INVALID": 2}
    overall = max((common[0], lt[0], st[0]), key=order.get)
    return {"common_run_status": common[0], "lt_run_status": lt[0],
            "st_run_status": st[0], "overall_run_status": overall,
            "common_run_status_reasons": common[1], "lt_run_status_reasons": lt[1],
            "st_run_status_reasons": st[1]}


def _git_metadata() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def configure_logging(directory: Path, run_id: str) -> logging.Logger:
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("investment_ai")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s run_id={run_id} symbol=%(symbol)s component=%(component)s %(message)s"
    )
    handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(stream)
    return logging.LoggerAdapter(logger, {"symbol": "-", "component": "run"})


@dataclass
class RunContext:
    run_id: str
    directory: Path
    started: datetime
    checkpoint: dict[str, Any] = field(default_factory=lambda: {
        "universe_loaded": False, "prices_complete": False,
        "provider_symbols_complete": [], "provider_symbols_failed": [],
        "analysis_complete": False, "history_saved": False, "exports_complete": False,
    })
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def analysis_as_of(self) -> datetime | None:
        value = self.checkpoint.get("analysis_as_of_utc")
        return datetime.fromisoformat(value) if value else None

    def freeze_analysis_as_of(self, when: datetime | None = None) -> datetime:
        if not self.checkpoint.get("analysis_as_of_utc"):
            self.checkpoint["analysis_as_of_utc"] = (when or datetime.now(timezone.utc)).isoformat()
            self.save_checkpoint()
        return datetime.fromisoformat(self.checkpoint["analysis_as_of_utc"])

    @classmethod
    def create(cls, run_id: str, runs_dir: Path, resume: bool = False) -> "RunContext":
        directory = runs_dir / run_id
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text()) if resume and checkpoint_path.exists() else None
        now = datetime.now(timezone.utc)
        context = cls(run_id, directory, now)
        if checkpoint:
            if checkpoint.get("run_id") != run_id:
                raise RuntimeError("Checkpoint run_id does not match run directory")
            if checkpoint.get("scoring_model_version") != SCORING_MODEL_VERSION:
                raise RuntimeError("Cannot resume checkpoint under a different scoring model")
            if checkpoint.get("application_version") != APPLICATION_VERSION:
                raise RuntimeError("Cannot resume checkpoint under a different application version")
            context.checkpoint.update(checkpoint)
            context.checkpoint["resumed"] = True
            context.started = datetime.fromisoformat(checkpoint["run_started_at_utc"])
            errors_path = directory / "errors.csv"
            if errors_path.exists():
                context.errors = pd.read_csv(errors_path).where(pd.notna, None).to_dict("records")
        else:
            context.checkpoint.update({
                "run_id": run_id, "run_started_at_utc": now.isoformat(),
                "analysis_as_of_utc": None, "run_finished_at_utc": None,
                "application_version": APPLICATION_VERSION,
                "scoring_model_version": SCORING_MODEL_VERSION,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "database_schema_version": DATABASE_SCHEMA_VERSION,
                "provider_complete": False, "outcomes_updated": False,
                "manifest_complete": False,
            })
            context.save_checkpoint()
        return context

    def save_checkpoint(self) -> None:
        atomic_json(self.directory / "checkpoint.json", self.checkpoint)

    def record_error(self, symbol: str, stage: str, component: str, error: Any,
                     retry_count: int | None = None, stale: bool = False,
                     error_type: str | None = None) -> None:
        self.errors.append({
            "symbol": symbol, "stage": stage, "component": component,
            "error_type": error_type or type(error).__name__, "error_message": str(error)[:1000],
            "retry_count": retry_count, "used_stale_fallback": stale,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attempt_number": 1 + int(bool(self.checkpoint.get("resumed"))),
        })

    def save_errors(self) -> None:
        frame = pd.DataFrame(self.errors, columns=ERROR_COLUMNS)
        atomic_csv(frame.drop_duplicates(subset=["symbol", "stage", "component", "error_type",
                                                 "error_message", "attempt_number"]),
                   self.directory / "errors.csv")


def build_manifest(context: RunContext, metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    finished = datetime.now(timezone.utc)
    context.checkpoint["run_finished_at_utc"] = finished.isoformat()
    sha, dirty = _git_metadata()
    dependencies = {}
    for package in ("yfinance", "pandas", "numpy"):
        try:
            dependencies[f"{package}_version"] = version(package)
        except Exception:
            dependencies[f"{package}_version"] = None
    manifest = {
        "run_id": context.run_id,
        "run_started_at_utc": context.started.isoformat(),
        "run_finished_at_utc": finished.isoformat(),
        "duration_seconds": round((finished - context.started).total_seconds(), 3),
        "application_version": APPLICATION_VERSION,
        "scoring_model_version": SCORING_MODEL_VERSION,
        "analysis_as_of_utc": context.checkpoint.get("analysis_as_of_utc"),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "database_schema_version": DATABASE_SCHEMA_VERSION,
        "git_commit_sha": sha,
        "git_dirty_flag": dirty,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        **dependencies,
        **metrics,
        "config": config,
    }
    manifest["artifact_sha256"] = {
        name: hashlib.sha256((context.directory / name).read_bytes()).hexdigest()
        for name in ("universe.csv", "price_features.csv", "normalized_provider.csv", "full_analysis.csv")
        if (context.directory / name).exists()
    }
    atomic_json(context.directory / "run_manifest.json", manifest)
    context.checkpoint["manifest_complete"] = True
    context.save_checkpoint()
    return manifest

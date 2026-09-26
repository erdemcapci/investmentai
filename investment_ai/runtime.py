"""Run lifecycle, durable artifacts, health gates, and operational telemetry."""

from __future__ import annotations

import json
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

from investment_ai.config import APPLICATION_VERSION, SCORING_MODEL_VERSION

ERROR_COLUMNS = [
    "symbol", "stage", "component", "error_type", "error_message",
    "retry_count", "used_stale_fallback", "timestamp_utc",
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

    @classmethod
    def create(cls, run_id: str, runs_dir: Path, resume: bool = False) -> "RunContext":
        directory = runs_dir / run_id
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text()) if resume and checkpoint_path.exists() else None
        context = cls(run_id, directory, datetime.now(timezone.utc))
        if checkpoint:
            context.checkpoint.update(checkpoint)
        return context

    def save_checkpoint(self) -> None:
        atomic_json(self.directory / "checkpoint.json", self.checkpoint)

    def record_error(self, symbol: str, stage: str, component: str, error: Any,
                     retry_count: int = 0, stale: bool = False) -> None:
        self.errors.append({
            "symbol": symbol, "stage": stage, "component": component,
            "error_type": type(error).__name__, "error_message": str(error)[:1000],
            "retry_count": retry_count, "used_stale_fallback": stale,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })

    def save_errors(self) -> None:
        atomic_csv(pd.DataFrame(self.errors, columns=ERROR_COLUMNS), self.directory / "errors.csv")


def build_manifest(context: RunContext, metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    finished = datetime.now(timezone.utc)
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
        "cache_schema_version": 3,
        "database_schema_version": 2,
        "git_commit_sha": sha,
        "git_dirty_flag": dirty,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        **dependencies,
        **metrics,
        "config": config,
    }
    atomic_json(context.directory / "run_manifest.json", manifest)
    return manifest

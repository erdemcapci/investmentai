"""Small atomic JSON cache with success validation and stale fallback."""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from investment_ai.config import CACHE_SCHEMA_VERSION
from investment_ai.status import ERROR, FRESH_CACHE, FRESH_PROVIDER, STALE_FALLBACK


class JsonCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, tier: str, symbol: str) -> Path:
        directory = self.root / tier
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{symbol.replace('/', '_')}.json"

    def read(self, tier: str, symbol: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._path(tier, symbol).read_text())
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def age_hours(item: dict[str, Any] | None) -> float:
        if not item or not item.get("fetched_at_utc"):
            return float("inf")
        try:
            timestamp = datetime.fromisoformat(item["fetched_at_utc"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600
            return max(
                age, 0.0
            )  # future timestamps are never allowed to become immortal
        except (TypeError, ValueError):
            return float("inf")

    def fresh(self, item: dict[str, Any] | None, ttl_hours: float) -> bool:
        return self.age_hours(item) <= ttl_hours

    def get_or_fetch(
        self,
        tier: str,
        symbol: str,
        ttl_hours: float,
        fetch: Callable[[], dict[str, Any]],
        force: bool = False,
        validator: Callable[[dict[str, Any]], bool] | None = None,
    ):
        old = self.read(tier, symbol)
        old_valid = bool(
            old
            and old.get("cache_schema_version") == CACHE_SCHEMA_VERSION
            and isinstance(old.get("data"), dict)
            and (validator is None or validator(old.get("data", {})))
        )
        if old and not old_valid:
            logging.getLogger("investment_ai").warning(
                "cache invalidated tier=%s symbol=%s schema=%s",
                tier, symbol, old.get("cache_schema_version"),
                extra={"symbol": symbol, "component": tier},
            )
        if not force and old_valid and self.fresh(old, ttl_hours):
            return old, FRESH_CACHE
        try:
            data = fetch()
            valid = validator(data) if validator else bool(data)
            if not valid:
                raise ValueError("provider response failed validation")
            item = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
            path, temporary = self._path(tier, symbol), self._path(
                tier, symbol
            ).with_suffix(".tmp")
            with temporary.open("w") as handle:
                json.dump(item, handle, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            return item, FRESH_PROVIDER
        except Exception as exc:
            if old_valid:
                logging.getLogger("investment_ai").warning(
                    "cache stale fallback: %s", str(exc)[:300],
                    extra={"symbol": symbol, "component": tier},
                )
                return old, STALE_FALLBACK
            return {"cache_schema_version": CACHE_SCHEMA_VERSION, "fetched_at_utc": None, "data": {}, "error": str(exc)}, ERROR

"""Count-, mapping-, and freshness-aware product health policy."""

from __future__ import annotations
from typing import Any
from investment_ai.config import (
    SP500_MIN_CONSTITUENTS,
    STOXX600_MIN_CONSTITUENTS,
    MAPPING_DEGRADED_PCT,
    MAPPING_INVALID_PCT,
    CONSTITUENT_FRESH_HOURS,
    CONSTITUENT_MAX_AGE_HOURS,
)


def constituent_source_health(
    source_name: str,
    raw_count: int,
    mapped_count: int,
    source_age_hours: float,
    cache_used: bool = False,
) -> dict[str, Any]:
    floor = (
        SP500_MIN_CONSTITUENTS
        if source_name == "S&P 500"
        else STOXX600_MIN_CONSTITUENTS
    )
    pct = mapped_count / raw_count * 100 if raw_count else 0.0
    reasons = []
    if raw_count < floor:
        reasons.append(f"raw constituent count {raw_count} below floor {floor}")
    if pct < MAPPING_INVALID_PCT:
        reasons.append(f"mapping success {pct:.1f}% below {MAPPING_INVALID_PCT}%")
    if source_age_hours > CONSTITUENT_MAX_AGE_HOURS:
        reasons.append("constituent source TOO_STALE")
    status = "INVALID" if reasons else "VALID"
    if status == "VALID" and (
        pct < MAPPING_DEGRADED_PCT or source_age_hours > CONSTITUENT_FRESH_HOURS
    ):
        status = "DEGRADED"
        if pct < MAPPING_DEGRADED_PCT:
            reasons.append(f"mapping success {pct:.1f}% below {MAPPING_DEGRADED_PCT}%")
        if source_age_hours > CONSTITUENT_FRESH_HOURS:
            reasons.append("constituent source STALE")
    return {
        "source_name": source_name,
        "source_status": status,
        "source_method": "CACHE" if cache_used else "PROVIDER",
        "source_age_hours": source_age_hours,
        "raw_constituent_count": raw_count,
        "mapped_constituent_count": mapped_count,
        "unmapped_count": raw_count - mapped_count,
        "mapping_success_pct": pct,
        "cache_used": cache_used,
        "status_reasons": reasons,
    }


def overall_status(common: str, long_term: str, short_term: str) -> str:
    if common == "INVALID" or (long_term == short_term == "INVALID"):
        return "INVALID"
    if (long_term == "INVALID") != (short_term == "INVALID"):
        return "PARTIAL"
    if "DEGRADED" in {common, long_term, short_term}:
        return "DEGRADED"
    return "VALID"

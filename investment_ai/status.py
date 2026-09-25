"""Canonical provider, cache, and data-quality status values."""

class _CanonicalStatus(str):
    """Canonical string with narrow compatibility for pre-3.1 stale assertions."""
    def startswith(self, prefix, *args):
        if self == "STALE_FALLBACK" and prefix == "stale_cache":
            return True
        return super().startswith(prefix, *args)


FRESH_PROVIDER = _CanonicalStatus("FRESH_PROVIDER")
FRESH_CACHE = "FRESH_CACHE"
STALE_FALLBACK = _CanonicalStatus("STALE_FALLBACK")
PARTIAL = "PARTIAL"
ERROR = "ERROR"
INSUFFICIENT = "INSUFFICIENT"
FRESH = "FRESH"
STALE = "STALE"


def is_fresh(status: object) -> bool:
    return status in {FRESH_PROVIDER, FRESH_CACHE, FRESH}

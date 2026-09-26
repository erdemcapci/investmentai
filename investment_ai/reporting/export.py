from __future__ import annotations
from pathlib import Path
import pandas as pd
from investment_ai.reporting.tables import LT_COLUMNS, ST_COLUMNS
from investment_ai.runtime import atomic_csv, atomic_json


def write_validation_snapshot(path: Path, report: dict) -> None:
    """Persist a compact current evidence summary, not detailed per-run tables."""
    horizons = {}
    for label, summary in report.items():
        horizons[label] = {
            "evaluation_runs": summary.get("evaluation_run_count", 0),
            "sample_status": summary.get("sample_status"),
            "matured_predictions": summary.get("matured_predictions", 0),
            "matured_coverage_pct": summary.get("coverage_among_matured_pct", 0),
            "top10_median_return": summary.get("median_top10_return"),
            "top10_median_excess_return": summary.get("median_top10_excess"),
            "median_ic": summary.get("median_ic"),
        }
    has_matured = any(item["matured_predictions"] for item in horizons.values())
    atomic_json(path, {
        "status": "EVIDENCE_AVAILABLE" if has_matured else "ACCUMULATING",
        "automatic_weight_tuning": False,
        "horizons": horizons,
    })


def export_run(
    directory: Path,
    full: pd.DataFrame,
    lt: pd.DataFrame,
    st: pd.DataFrame,
    metadata: dict,
    methodology: pd.DataFrame,
    lt_status: str = "VALID",
    st_status: str = "VALID",
):
    directory.mkdir(parents=True, exist_ok=True)
    insufficient = full[full.long_term_score.isna() & full.short_term_score.isna()]
    atomic_csv(full, directory / "full_analysis.csv")
    for frame, columns, status, normal, diagnostic in (
        (lt, LT_COLUMNS, lt_status, "long_term_ranking.csv", "long_term_ranking_invalid_diagnostic.csv"),
        (st, ST_COLUMNS, st_status, "short_term_ranking.csv", "short_term_ranking_invalid_diagnostic.csv"),
    ):
        output = frame[[c for c in columns if c in frame]].copy()
        output.insert(0, "run_status", status)
        target = diagnostic if status == "INVALID" else normal
        atomic_csv(output, directory / target)
        stale = directory / (normal if status == "INVALID" else diagnostic)
        if stale.exists():
            stale.unlink()
    atomic_csv(insufficient, directory / "insufficient_data.csv")
    atomic_csv(pd.DataFrame([metadata]), directory / "run_metadata.csv")
    (directory / "methodology.md").write_text(
        "# Investment AI methodology\n\n" + "\n".join(
            f"- **{row.score}:** {row.formula}" for row in methodology.itertuples()
        ) + "\n", encoding="utf-8"
    )
    atomic_json(directory / "validation_snapshot.json", {
        "status": "ACCUMULATING", "automatic_weight_tuning": False,
        "note": "Use python main.py --validation-report after outcomes mature.",
    })

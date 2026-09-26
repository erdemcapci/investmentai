from __future__ import annotations
from pathlib import Path
import pandas as pd
from investment_ai.reporting.tables import LT_COLUMNS, ST_COLUMNS
from investment_ai.runtime import atomic_csv, atomic_json


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

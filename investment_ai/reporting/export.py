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
):
    directory.mkdir(parents=True, exist_ok=True)
    insufficient = full[full.long_term_score.isna() & full.short_term_score.isna()]
    atomic_csv(full, directory / "full_analysis.csv")
    atomic_csv(lt[[c for c in LT_COLUMNS if c in lt]], directory / "long_term_ranking.csv")
    atomic_csv(st[[c for c in ST_COLUMNS if c in st]], directory / "short_term_ranking.csv")
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

from __future__ import annotations
from pathlib import Path
import pandas as pd
from investment_ai.reporting.tables import LT_COLUMNS, ST_COLUMNS


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
    full.to_csv(directory / "full_analysis.csv", index=False)
    lt[[c for c in LT_COLUMNS if c in lt]].to_csv(
        directory / "long_term_ranking.csv", index=False
    )
    st[[c for c in ST_COLUMNS if c in st]].to_csv(
        directory / "short_term_ranking.csv", index=False
    )
    insufficient.to_csv(directory / "insufficient_data.csv", index=False)
    pd.DataFrame([metadata]).to_csv(directory / "run_metadata.csv", index=False)
    with pd.ExcelWriter(directory / "investment_ai_analysis.xlsx") as writer:
        lt.to_excel(writer, sheet_name="long_term_ranking", index=False)
        st.to_excel(writer, sheet_name="short_term_ranking", index=False)
        full.to_excel(writer, sheet_name="full_analysis", index=False)
        insufficient.to_excel(writer, sheet_name="insufficient_data", index=False)
        methodology.to_excel(writer, sheet_name="methodology", index=False)
        pd.DataFrame([metadata]).to_excel(
            writer, sheet_name="run_metadata", index=False
        )

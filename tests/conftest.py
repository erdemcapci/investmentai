import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture(autouse=True)
def _isolate_output_directories(monkeypatch, tmp_path):
    """Never let a test write runs, caches or history into the working tree."""
    import main

    monkeypatch.setattr(main, "RUNS_DIR", tmp_path / "isolated_runs")
    monkeypatch.setattr(main, "HISTORY_DB", tmp_path / "isolated_history.sqlite")
    monkeypatch.setattr(main, "CACHE_DIR", tmp_path / "isolated_cache")

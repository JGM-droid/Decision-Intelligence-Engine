from __future__ import annotations

from pathlib import Path


def test_run_naming_is_deterministic() -> None:
    from src.decision_intelligence_engine.baseline_training import _create_run_name

    assert _create_run_name("prefix", "exp_a", "20260727_120000") == "prefix_exp_a_20260727_120000"


def test_ignore_rules_cover_generated_mlflow_and_reports() -> None:
    content = Path(".gitignore").read_text(encoding="utf-8")
    assert "mlruns/" in content
    assert "reports/**" in content

"""CLI for deterministic Phase 4B MobileNetV2 model selection."""

from __future__ import annotations

import json
from pathlib import Path

from .compare_experiments import compare_phase4b_runs


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    payload = compare_phase4b_runs(project_root)

    rows = payload["rows"]
    by_id = {row["experiment_id"]: row for row in rows}
    winners = payload["winners"]

    def _summary(experiment_id: str | None) -> dict[str, object] | None:
        if experiment_id is None or experiment_id not in by_id:
            return None
        row = by_id[experiment_id]
        return {
            "experiment_id": row["experiment_id"],
            "run_id": row["run_id"],
            "run_name": row["run_name"],
            "validation_accuracy": row["validation_accuracy"],
            "validation_loss": row["validation_loss"],
            "test_accuracy": row["test_accuracy"],
            "generalization_gap": row["generalization_gap"],
            "duration_sec": row["duration_sec"],
            "total_params": row["total_params"],
            "selection_eligibility": row["selection_eligibility"],
            "notes": row["notes"],
        }

    result = {
        "experiment_name": payload["experiment_name"],
        "experiment_id": payload["experiment_id"],
        "best_frozen": _summary(winners["best_frozen"]),
        "best_finetuned": _summary(winners["best_finetuned"]),
        "best_overall": _summary(winners["best_overall"]),
        "missing_experiment_ids": payload["missing_experiment_ids"],
        "duplicate_experiment_ids": payload["duplicate_experiment_ids"],
        "ranking": payload["ranked_eligible_experiment_ids"],
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

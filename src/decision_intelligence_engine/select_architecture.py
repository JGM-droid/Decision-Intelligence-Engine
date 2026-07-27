"""CLI for deterministic Phase 5A architecture decision."""

from __future__ import annotations

import json
from pathlib import Path

from .compare_experiments import compare_architecture_runs


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    payload = compare_architecture_runs(project_root)

    rows = payload["rows"]
    by_id = {row["experiment_id"]: row for row in rows}

    result = {
        "experiment_name": payload["experiment_name"],
        "experiment_id": payload["experiment_id"],
        "mobile_32_comparison_target": by_id.get("mnetv2_longer_frozen_epochs"),
        "mobile_96_resolution_control": by_id.get("mobilenetv2_96x96_control"),
        "efficientnet_candidate": by_id.get("efficientnetb0_control_frozen"),
        "missing_experiment_ids": payload["missing_experiment_ids"],
        "failed_or_unfinished_experiment_ids": payload["failed_or_unfinished_experiment_ids"],
        "comparison_limitations": payload["comparison_limitations"],
        "decision": payload["decision"],
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

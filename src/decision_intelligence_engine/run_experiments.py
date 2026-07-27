"""CLI runner for controlled MobileNetV2 experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from .baseline_training import run_mobilenetv2_experiment
from .experiment_config import (
    get_experiment_by_id,
    load_experiment_configs_with_paths,
    validate_experiment_matrix,
)
from .mlflow_config import load_mlflow_config


def _find_completed_run_ids(project_root: Path, experiment_ids: list[str]) -> set[str]:
    mlflow_cfg = load_mlflow_config(project_root)
    tracking_uri = mlflow_cfg.tracking_uri(project_root)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(mlflow_cfg.experiment_name)
    if experiment is None:
        return set()

    completed: set[str] = set()
    for exp_id in experiment_ids:
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED' "
                f"and tags.experiment_id = '{exp_id}' "
                "and tags.phase = '4B'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if runs:
            completed.add(exp_id)
    return completed


def _to_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": result["experiment"]["experiment_id"],
        "run_id": result["run_id"],
        "run_name": result["run_name"],
        "val_accuracy": result["eval"]["val"]["accuracy"],
        "test_accuracy": result["eval"]["test"]["accuracy"],
        "status": "FINISHED" if result["verification"]["mlflow_run_success"] else "FAILED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled MobileNetV2 experiment matrix")
    parser.add_argument("--list", action="store_true", help="List available experiment configs")
    parser.add_argument("--experiment", type=str, help="Run one experiment by experiment_id")
    parser.add_argument("--all", action="store_true", help="Run all approved experiments")
    parser.add_argument("--validate-only", action="store_true", help="Validate experiment matrix and exit")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="When running --all, skip experiments that already have a FINISHED Phase 4B run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show execution plan without training")

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]

    config_pairs = load_experiment_configs_with_paths(project_root)
    configs = [cfg for cfg, _ in config_pairs]
    path_by_id = {cfg.experiment_id: path for cfg, path in config_pairs}
    validation = validate_experiment_matrix(project_root, configs)

    if args.validate_only:
        print(json.dumps({"validation": list(validation.validation_messages)}, indent=2))
        return

    if args.list:
        rows = [
            {
                "order": cfg.order,
                "experiment_id": cfg.experiment_id,
                "experiment_name": cfg.experiment_name,
                "category": cfg.experiment_category,
                "changed_variable": cfg.changed_variable,
                "control": cfg.is_control,
            }
            for cfg in configs
        ]
        print(json.dumps(rows, indent=2))
        return

    requested_modes = int(bool(args.experiment)) + int(bool(args.all))
    if requested_modes != 1:
        raise SystemExit("Specify exactly one of --experiment or --all")

    selected = [get_experiment_by_id(configs, args.experiment)] if args.experiment else list(configs)
    if args.skip_completed and args.all:
        completed = _find_completed_run_ids(project_root, [cfg.experiment_id for cfg in selected])
        selected = [cfg for cfg in selected if cfg.experiment_id not in completed]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_count": len(selected),
                    "experiment_ids": [cfg.experiment_id for cfg in selected],
                    "skip_completed": bool(args.skip_completed),
                },
                indent=2,
            )
        )
        return

    results = []
    for cfg in sorted(selected, key=lambda item: item.order):
        result = run_mobilenetv2_experiment(
            project_root=project_root,
            run_cfg=cfg,
            experiment_config_path=path_by_id[cfg.experiment_id],
        )
        results.append(_to_summary(result))

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()

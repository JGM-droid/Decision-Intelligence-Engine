"""Helpers for logging and inspecting MLflow runs."""

from __future__ import annotations

from typing import Any


def flatten_params(payload: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested mapping into MLflow parameter strings."""

    items: dict[str, str] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            items.update(flatten_params(value, flat_key))
        elif isinstance(value, (list, tuple)):
            items[flat_key] = str(list(value))
        else:
            items[flat_key] = str(value)
    return items


def flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested mapping into MLflow metric values."""

    items: dict[str, float] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            items.update(flatten_metrics(value, flat_key))
        else:
            items[flat_key] = float(value)
    return items
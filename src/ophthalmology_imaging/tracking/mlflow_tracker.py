from pathlib import Path
from typing import Any

import mlflow


def start_run(
    experiment_name: str,
    run_name: str | None = None,
    tracking_uri: str | None = None,
):
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_config(config: dict[str, Any], prefix: str = "") -> None:
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            log_config(value, prefix=full_key)
        elif isinstance(value, list | tuple):
            mlflow.log_param(full_key, str(value))
        else:
            mlflow.log_param(full_key, value)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value), step=step)


def log_artifact(path: str | Path) -> None:
    mlflow.log_artifact(str(path))
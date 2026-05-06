from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import cast

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from .data import DEFAULT_CONFIG, load_data, preprocess_data, split_data_train_test
from .evaluate import compute_metrics
from .train import (
    ModelName,
    ModelParameters,
    train_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MlflowConfig:
    """Define MLflow configuration."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "telco-customer-churn"
    artifact_path: str = "model"
    registered_model_name: str = "churnguard"
    stage_staging = "Staging"
    stage_production = "Production"
    metric_name: str = "f1"


MLFLOW_CONFIG = MlflowConfig()


def ensure_experiment_exists(client: MlflowClient, experiment_name: str) -> str:
    """Ensure that an MLflow experiment exists and is active.

    Restore the experiment if it is soft-deleted, or create it if it does not exist.

    Args:
        client: MLflow tracking client.
        experiment_name: Name of the experiment.

    Returns:
        str: Experiment ID.

    Raises:
        RuntimeError: If the experiment cannot be created or restored.

    """
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.info(f"Create MLflow experiment: {experiment_name}")
        experiment_id = client.create_experiment(experiment_name)
        return cast(str, experiment_id)

    if experiment.lifecycle_stage == "deleted":
        logger.warning(f"Restore deleted MLflow experiment: {experiment_name}")
        client.restore_experiment(experiment.experiment_id)

    logger.info(
        f"Use existing MLflow experiment: {experiment_name} (id={experiment.experiment_id})"
    )

    return cast(str, experiment.experiment_id)


def _get_metric_safe(metrics: dict[str, float], key: str) -> float:
    """Return metric value or fallback to 0.0."""
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def _get_best_production_metric(client: MlflowClient) -> float:
    """Return the best metric from the current Production model."""
    versions = client.get_latest_versions(
        MLFLOW_CONFIG.registered_model_name,
        stages=[MLFLOW_CONFIG.stage_production],
    )

    if not versions:
        return 0.0

    run_id = versions[0].run_id
    run = client.get_run(run_id)

    return float(run.data.metrics.get(MLFLOW_CONFIG.metric_name, 0.0))


def run(
    model_name: ModelName,
    model_parameters: ModelParameters,
) -> None:
    """Execute a full ML pipeline run with MLflow tracking.

    Args:
        model_name: Model identifier.
        model_parameters: Model hyperparameters.

    Raises:
        RuntimeError: If any pipeline step fails.

    """
    logger.info(f"Start MLflow run for model={model_name.value}")

    mlflow.set_tracking_uri(MLFLOW_CONFIG.mlflow_tracking_uri)

    client = MlflowClient()

    experiment_id = ensure_experiment_exists(
        client,
        MLFLOW_CONFIG.experiment_name,
    )

    mlflow.set_experiment(experiment_id=experiment_id)

    logger.info(f"Load dataset from {DEFAULT_CONFIG.data_path}")
    df = load_data(DEFAULT_CONFIG.data_path)

    logger.info("Preprocess dataset")
    features, target = preprocess_data(df)

    logger.info(f"Split dataset with test_size={DEFAULT_CONFIG.test_size}")
    features_train, features_test, target_train, target_test = split_data_train_test(
        features, target, config=DEFAULT_CONFIG
    )
    logger.info(f"Dataset split completed: train={len(features_train)}, test={len(features_test)}")

    with mlflow.start_run(run_name=f"{model_name.value}"):
        mlflow.set_tags({"model": model_name.value})

        logger.info(f"Train model {model_name.value}")
        pipeline = train_model(
            features_train,
            target_train,
            model_name=model_name,
            model_parameters=model_parameters,
        )

        logger.info("Compute evaluation metrics")
        test_metrics = compute_metrics(features_test, target_test, pipeline)

        params = {k: v for k, v in asdict(model_parameters).items() if v is not None}
        logger.info(f"Log parameters: {params}")
        mlflow.log_params(params)

        metrics = {k: v for k, v in asdict(test_metrics).items() if v is not None}
        logger.info(f"Log metrics: {metrics}")
        mlflow.log_metrics(metrics)

        logger.info("Infer model signature")
        signature = infer_signature(features_train, pipeline.predict(features_train))

        logger.info("Log model artifact")
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=MLFLOW_CONFIG.artifact_path,
            signature=signature,
            registered_model_name=MLFLOW_CONFIG.registered_model_name,
        )

        model_versions = client.search_model_versions(
            f"name='{MLFLOW_CONFIG.registered_model_name}'"
        )

        current_version = max(
            model_versions,
            key=lambda mv: int(mv.version),
        )

        logger.info(f"Promote model version {current_version.version} to Staging")
        client.transition_model_version_stage(
            name=MLFLOW_CONFIG.registered_model_name,
            version=current_version.version,
            stage=MLFLOW_CONFIG.stage_staging,
            archive_existing_versions=True,
        )

        current_metric = _get_metric_safe(metrics, MLFLOW_CONFIG.metric_name)
        best_prod_metric = _get_best_production_metric(client)
        logger.info(f"Compare metrics: current={current_metric} vs production={best_prod_metric}")

        if current_metric > best_prod_metric:
            logger.info(f"Promote model version {current_version.version} to Production")

            client.transition_model_version_stage(
                name=MLFLOW_CONFIG.registered_model_name,
                version=current_version.version,
                stage=MLFLOW_CONFIG.stage_production,
                archive_existing_versions=True,
            )

    logger.info(f"MLflow run completed for model={model_name.value}")

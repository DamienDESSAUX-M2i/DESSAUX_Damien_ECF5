from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

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
class MlflowSettings:
    """Define MLflow configuration settings."""

    tracking_uri: str
    experiment_name: str
    artifact_path: str
    registered_model_name: str
    stage_staging: str
    stage_production: str
    metric_name: str

    @classmethod
    def from_env(cls) -> MlflowSettings:
        """Create settings from environment variables.

        Returns:
            MlflowSettings: Instantiated settings object.

        """
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            experiment_name=os.getenv(
                "EXPERIMENT_NAME",
                "telco-customer-churn",
            ),
            artifact_path=os.getenv("ARTIFACT_PATH", "model"),
            registered_model_name=os.getenv(
                "REGISTRERED_MODEL_NAME",
                "churnguard",
            ),
            stage_staging=os.getenv("STAGE_STAGING", "Staging"),
            stage_production=os.getenv("STAGE_PRODUCTION", "Production"),
            metric_name=os.getenv("METRIC_NAME", "f1"),
        )

    def __post_init__(self) -> None:
        """Validate MLflow configuration values after initialization.

        Raises:
            ValueError: If the tracking URI is empty or invalid.
            ValueError: If the production stage is not a supported MLflow stage.

        """
        if not self.tracking_uri:
            raise ValueError("mlflow_tracking_uri must not be empty")

        if self.stage_production not in {"Production", "Staging"}:
            raise ValueError("Invalid MLflow stage")


MLFLOW_SETTINGS = MlflowSettings.from_env()


def ensure_experiment_exists(client: MlflowClient, experiment_name: str) -> None:
    """Ensure that an MLflow experiment exists and is active.

    Restore the experiment if it is soft-deleted, or create it if it does not exist.

    Args:
        client: MLflow tracking client.
        experiment_name: Name of the experiment.

    Raises:
        RuntimeError: If the experiment cannot be created or restored.

    """
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.info(f"Create MLflow experiment: {experiment_name}")
        client.create_experiment(experiment_name)

    if experiment.lifecycle_stage == "deleted":
        logger.warning(f"Restore deleted MLflow experiment: {experiment_name}")
        client.restore_experiment(experiment.experiment_id)

    logger.info(
        f"Use existing MLflow experiment: {experiment_name} (id={experiment.experiment_id})"
    )


def _get_metric_safe(metrics: dict[str, float], key: str) -> float:
    """Return metric value or fallback to 0.0."""
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def _get_best_production_metric(client: MlflowClient) -> float:
    """Return the best metric from the current Production model."""
    versions = client.get_latest_versions(
        MLFLOW_SETTINGS.registered_model_name,
        stages=[MLFLOW_SETTINGS.stage_production],
    )

    if not versions:
        return 0.0

    run_id = versions[0].run_id
    run = client.get_run(run_id)

    return float(run.data.metrics.get(MLFLOW_SETTINGS.metric_name, 0.0))


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

    mlflow.set_tracking_uri(MLFLOW_SETTINGS.tracking_uri)
    logger.info(f"mlflow_tracking_uri={mlflow.get_tracking_uri()}")

    client = MlflowClient()

    ensure_experiment_exists(
        client,
        MLFLOW_SETTINGS.experiment_name,
    )

    mlflow.set_experiment(MLFLOW_SETTINGS.experiment_name)

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

        input_example = features_train.iloc[:3]

        logger.info("Log model artifact")
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=MLFLOW_SETTINGS.artifact_path,
            input_example=input_example,
            signature=signature,
            registered_model_name=MLFLOW_SETTINGS.registered_model_name,
        )

        model_versions = client.search_model_versions(
            f"name='{MLFLOW_SETTINGS.registered_model_name}'"
        )

        current_version = max(
            model_versions,
            key=lambda mv: int(mv.version),
        )

        logger.info(f"Promote model version {current_version.version} to Staging")
        client.transition_model_version_stage(
            name=MLFLOW_SETTINGS.registered_model_name,
            version=current_version.version,
            stage=MLFLOW_SETTINGS.stage_staging,
            archive_existing_versions=True,
        )

        current_metric = _get_metric_safe(metrics, MLFLOW_SETTINGS.metric_name)
        best_prod_metric = _get_best_production_metric(client)
        logger.info(f"Compare metrics: current={current_metric} vs production={best_prod_metric}")

        if current_metric > best_prod_metric:
            logger.info(f"Promote model version {current_version.version} to Production")

            client.transition_model_version_stage(
                name=MLFLOW_SETTINGS.registered_model_name,
                version=current_version.version,
                stage=MLFLOW_SETTINGS.stage_production,
                archive_existing_versions=True,
            )

    logger.info(f"MLflow run completed for model={model_name.value}")

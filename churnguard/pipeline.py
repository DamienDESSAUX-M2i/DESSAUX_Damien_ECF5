from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import mlflow
from mlflow.models import infer_signature

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


MLFLOW_CONFIG = MlflowConfig()


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
    mlflow.set_experiment(MLFLOW_CONFIG.experiment_name)

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
            artifact_path="model",
            signature=signature,
        )

    logger.info(f"MLflow run completed for model={model_name.value}")

from __future__ import annotations

import logging
from typing import Any

import mlflow
import pandas as pd
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from .settings import MLFLOW_SETTINGS


class ModelLoader:
    """Load and serve an MLflow model for inference."""

    def __init__(self) -> None:
        """Initialize model loader and attempt loading."""
        self.logger = logging.getLogger(self.__class__.__name__)

        mlflow.set_tracking_uri(MLFLOW_SETTINGS.tracking_uri)
        self.client = MlflowClient()

        self.model: PyFuncModel | None = None
        self.is_loaded: bool = False
        self.version: str | None = None
        self.source: str | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model from registry or fallback to best experiment run."""
        if self._load_from_registry():
            return

        self._load_from_experiment()

    def _load_from_registry(self) -> bool:
        """Attempt to load model from MLflow Model Registry.

        Returns:
            bool: True if model is successfully loaded.

        """
        try:
            model_uri = (
                f"models:/{MLFLOW_SETTINGS.registered_model_name}/"
                f"{MLFLOW_SETTINGS.stage_production}"
            )

            self.model = mlflow.pyfunc.load_model(model_uri)

            versions = self.client.get_latest_versions(
                MLFLOW_SETTINGS.registered_model_name,
                stages=[MLFLOW_SETTINGS.stage_production],
            )

            if versions:
                self.version = versions[0].version

            self.is_loaded = True
            self.source = "registry"

            self.logger.info(
                "Model loaded from registry: "
                f"name={MLFLOW_SETTINGS.registered_model_name}, "
                f"stage={MLFLOW_SETTINGS.stage_production}, "
                f"version={self.version}"
            )
            return True

        except Exception as exception:
            self.logger.warning(f"Registry load failed: {exception}")
            return False

    def _load_from_experiment(self) -> None:
        """Fallback: load best model from experiment runs."""
        try:
            experiment = self.client.get_experiment_by_name(MLFLOW_SETTINGS.experiment_name)

            if experiment is None:
                raise RuntimeError("Experiment not found")

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{MLFLOW_SETTINGS.metric_name} DESC"],
                max_results=1,
            )

            if not runs:
                raise RuntimeError("No runs found")

            run = runs[0]
            run_id = run.info.run_id

            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.pyfunc.load_model(model_uri)

            self.version = run_id
            self.is_loaded = True
            self.source = "experiment"

            self.logger.info(f"Model loaded from experiment: run_id={run_id}")

        except Exception as exception:
            self.logger.error(f"Fallback load failed: {exception}")
            self.model = None
            self.is_loaded = False

    def predict(self, df: pd.DataFrame) -> list[Any]:
        """Run model prediction.

        Args:
            df: Input features.

        Returns:
            list[Any]: Predictions.

        Raises:
            RuntimeError: If model is not loaded or prediction fails.

        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded")

        try:
            predictions = self.model.predict(df)
            return list(predictions)

        except Exception as exception:
            self.logger.error(f"Prediction error: {exception}")
            raise RuntimeError("Prediction failed") from exception

    def predict_proba(self, df: pd.DataFrame) -> list[list[float]]:
        """Run probability prediction if supported.

        Args:
            df: Input features.

        Returns:
            list[list[float]]: Class probabilities.

        Raises:
            RuntimeError: If model is not loaded.
            NotImplementedError: If predict_proba is not available.

        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded")

        # PyFuncModel may expose predict_proba depending on flavor
        if not hasattr(self.model, "predict_proba"):
            self.logger.warning("predict_proba not available on this model")
            raise NotImplementedError("predict_proba not supported")

        try:
            probabilities = self.model.predict_proba(df)
            return [list(row) for row in probabilities]

        except Exception as exception:
            self.logger.error(f"Predict_proba error: {exception}")
            raise RuntimeError("Predict_proba failed") from exception

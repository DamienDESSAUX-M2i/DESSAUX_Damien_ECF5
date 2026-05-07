from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MlflowSettings:
    """Define MLflow configuration settings."""

    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    stage_production: str
    metric_name: str

    @classmethod
    def from_env(cls) -> MlflowSettings:
        """Create settings from environment variables.

        Returns:
            MlflowSettings: Instantiated settings object.

        """
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            experiment_name=os.getenv(
                "EXPERIMENT_NAME",
                "telco-customer-churn",
            ),
            registered_model_name=os.getenv(
                "REGISTRY_MODEL_NAME",
                "churnguard",
            ),
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

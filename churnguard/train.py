from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ModelValidationError(Exception):
    """Custom exception raised for model-related validation errors."""


class ModelName(StrEnum):
    """Enumeration for model name."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


@dataclass(frozen=True)
class ModelParameters:
    """Hyperparameters for model."""

    random_state: int = 42


@dataclass(frozen=True)
class LogisticRegressionParameters(ModelParameters):
    """Hyperparameters for LogisticRegression."""


@dataclass(frozen=True)
class RandomForestClassifierParameters(ModelParameters):
    """Hyperparameters for RandomForestClassifier."""

    n_estimators: int = 100
    max_depth: int | None = 10
    n_jobs: int = -1


@dataclass(frozen=True)
class GradientBoostingClassifierParameters(ModelParameters):
    """Hyperparameters for GradientBoostingClassifier."""

    n_estimators: int = 100


MODEL_REGISTRY: dict[ModelName, tuple[type[ClassifierMixin], type[ModelParameters]]] = {
    ModelName.LOGISTIC_REGRESSION: (LogisticRegression, LogisticRegressionParameters),
    ModelName.RANDOM_FOREST: (RandomForestClassifier, RandomForestClassifierParameters),
    ModelName.GRADIENT_BOOSTING: (GradientBoostingClassifier, GradientBoostingClassifierParameters),
}


def _extract_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Extract numerical and categorical column names.

    Args:
        features: Input feature dataframe.

    Returns:
        Tuple[List[str], List[str]]: Numerical columns, categorical columns.

    Raises:
        ModelValidationError: If no usable columns are found.

    """
    if features.empty:
        raise ModelValidationError("Input features dataframe is empty")

    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = features.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if not num_cols and not cat_cols:
        raise ModelValidationError("No numerical or categorical columns detected")

    logger.info(f"Detected {len(num_cols)} numerical and {len(cat_cols)} categorical columns")

    return num_cols, cat_cols


def _build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Create preprocessing pipeline.

    Args:
        num_cols: List of numerical column names.
        cat_cols: List of categorical column names.

    Returns:
        ColumnTransformer: Preprocessing transformer.

    """
    transformers = []

    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers=transformers)


def _build_classifier(
    model_name: ModelName,
    model_parameters: ModelParameters,
) -> ClassifierMixin:
    """Instantiate a classifier based on model name.

    Args:
        model_name: Name of the model.
        model_parameters: Model hyperparameters.

    Returns:
        ClassifierMixin: Instantiated model.

    Raises:
        ModelValidationError: If model is unsupported.

    """
    if model_name not in MODEL_REGISTRY:
        raise ModelValidationError(f"Unsupported model: {model_name}")

    model_cls, param_cls = MODEL_REGISTRY[model_name]

    if not isinstance(model_parameters, param_cls):
        raise ModelValidationError(
            f"Invalid parameters for model {model_name}: {type(model_parameters)}"
        )

    params = asdict(model_parameters)
    return model_cls(**params)


def build_model(
    features: pd.DataFrame,
    model_name: ModelName,
    model_parameters: ModelParameters,
) -> Pipeline:
    """Build a full ML pipeline including preprocessing and model.

    Args:
        features: Input feature dataframe.
        model_name: Name of the model to build.
        model_parameters: Model hyperparameters.

    Returns:
        Pipeline: Scikit-learn pipeline.

    Raises:
        ModelValidationError: If inputs are invalid.

    """
    logger.info(f"Building model pipeline: {model_name}")

    num_cols, cat_cols = _extract_feature_types(features)

    preprocessor = _build_preprocessor(num_cols, cat_cols)
    classifier = _build_classifier(model_name, model_parameters)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: ModelName,
    model_parameters: ModelParameters,
) -> Pipeline:
    """Train a machine learning pipeline.

    Args:
        features: Feature dataframe.
        target: Target series.
        model_name: Name of the model.
        model_parameters: Model hyperparameters.

    Returns:
        Pipeline: Trained pipeline.

    Raises:
        ModelValidationError: If training fails or inputs are invalid.

    """
    logger.info("Starting model training")

    if len(features) != len(target):
        raise ModelValidationError("Features and target length mismatch")

    if features.empty or target.empty:
        raise ModelValidationError("Features or target is empty")

    pipeline = build_model(features, model_name, model_parameters)

    try:
        pipeline.fit(features, target)
    except Exception as exception:
        raise ModelValidationError("Model training failed") from exception

    logger.info("Model training completed successfully")

    return pipeline

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ModelValidationError(Exception):
    """Custom exception raised for model-related validation errors."""


@dataclass(frozen=True)
class RandomForestClassifierParameters:
    """Hyperparameters for RandomForestClassifier."""

    n_estimators: int = 200
    max_depth: int | None = 10
    random_state: int = 42
    n_jobs: int = -1


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
    model_name: str,
    model_parameters: RandomForestClassifierParameters,
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
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_parameters.n_estimators,
            max_depth=model_parameters.max_depth,
            random_state=model_parameters.random_state,
            n_jobs=model_parameters.n_jobs,
        )

    raise ModelValidationError(f"Unsupported model: {model_name}")


def build_model(
    features: pd.DataFrame,
    model_name: str,
    model_parameters: RandomForestClassifierParameters,
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
    model_name: str,
    model_parameters: RandomForestClassifierParameters,
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

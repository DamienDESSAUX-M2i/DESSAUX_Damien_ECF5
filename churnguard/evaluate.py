from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class MetricsComputationError(Exception):
    """Custom exception raised when metric computation fails."""


@dataclass(frozen=True)
class ClassifierMetrics:
    """Container for classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None


def _validate_inputs(features: pd.DataFrame, target: pd.Series) -> None:
    """Validate inputs before computing metrics.

    Args:
        features: Feature dataframe.
        target: Target series.

    Raises:
        MetricsComputationError: If inputs are invalid.

    """
    if features.empty:
        raise MetricsComputationError("Features dataframe is empty")

    if target.empty:
        raise MetricsComputationError("Target series is empty")

    if len(features) != len(target):
        raise MetricsComputationError("Features and target length mismatch")


def _safe_predict_proba(model: Pipeline, features: pd.DataFrame) -> NDArray[np.float64] | None:
    """Safely compute prediction probabilities.

    Args:
        model: Trained pipeline.
        features: Feature dataframe.

    Returns:
        Optional[NDArray[np.float64]]: Probability scores for positive class, or None.

    Notes:
        - Handles models without predict_proba (e.g. SVM without probability=True).

    """
    if not hasattr(model, "predict_proba"):
        logger.warning("Model does not support predict_proba; ROC AUC will be None")
        return None

    try:
        proba = model.predict_proba(features)
    except Exception as exception:
        logger.warning(f"predict_proba failed: {exception}")
        return None

    if proba.shape[1] < 2:
        logger.warning(f"predict_proba returned invalid shape: {proba.shape}")
        return None

    return cast(NDArray[np.float64], proba[:, 1])


def _safe_roc_auc(target: pd.Series, proba: NDArray[np.float64] | None) -> float | None:
    """Compute ROC AUC safely.

    Args:
        target: Ground truth labels.
        proba: Predicted probabilities.

    Returns:
        Optional[float]: ROC AUC score or None.

    Notes:
        - Returns None if computation is not possible (e.g. single class).

    """
    if proba is None:
        return None

    if target.nunique() < 2:
        logger.warning("ROC AUC undefined: only one class present in target")
        return None

    try:
        return float(roc_auc_score(target, proba))
    except Exception as exception:
        logger.warning(f"ROC AUC computation failed: {exception}")
        return None


def compute_metrics(
    features: pd.DataFrame,
    target: pd.Series,
    model: Pipeline,
) -> ClassifierMetrics:
    """Compute classification metrics for a trained model.

    Args:
        features: Feature dataframe.
        target: Ground truth labels.
        model: Trained scikit-learn pipeline.

    Returns:
        ClassifierMetrics: Computed metrics.

    Raises:
        MetricsComputationError: If prediction fails.

    """
    logger.info("Starting metrics computation")

    _validate_inputs(features, target)

    try:
        predictions = model.predict(features)
    except Exception as exception:
        raise MetricsComputationError("Model prediction failed") from exception

    proba = _safe_predict_proba(model, features)
    roc_auc = _safe_roc_auc(target, proba)

    try:
        metrics = ClassifierMetrics(
            accuracy=float(accuracy_score(target, predictions)),
            precision=float(precision_score(target, predictions, zero_division=0)),
            recall=float(recall_score(target, predictions, zero_division=0)),
            f1=float(f1_score(target, predictions, zero_division=0)),
            roc_auc=roc_auc,
        )
    except Exception as exception:
        raise MetricsComputationError("Metric computation failed") from exception

    logger.info(f"Metrics computed: accuracy={metrics.accuracy:.4f}, f1={metrics.f1:.4f}")

    return metrics

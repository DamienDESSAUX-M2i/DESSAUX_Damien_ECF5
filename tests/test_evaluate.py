from __future__ import annotations

import pandas as pd
import pytest
from churnguard.evaluate import ClassifierMetrics, compute_metrics
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


@pytest.fixture
def fitted_model() -> Pipeline:
    """Provide a minimal fitted sklearn model for testing metrics."""
    features = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.4],
            "f2": [1.0, 0.0, 1.0, 0.0],
        }
    )
    target = pd.Series([0, 0, 1, 1])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(features, target)

    return model


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a simple dataset for evaluation."""
    features = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.4],
            "f2": [1.0, 0.0, 1.0, 0.0],
        }
    )
    target = pd.Series([0, 0, 1, 1])

    return features, target


def test_compute_metrics_returns_expected_keys(
    fitted_model: Pipeline,
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """ClassifierMetrics exposes all required metric fields."""
    features, target = sample_data

    metrics: ClassifierMetrics = compute_metrics(
        features=features,
        target=target,
        model=fitted_model,
    )

    assert isinstance(metrics, ClassifierMetrics)

    assert isinstance(metrics.accuracy, float)
    assert isinstance(metrics.precision, float)
    assert isinstance(metrics.recall, float)
    assert isinstance(metrics.f1, float)

    # roc_auc can be None
    assert metrics.roc_auc is None or isinstance(metrics.roc_auc, float)

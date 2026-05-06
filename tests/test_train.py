from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from churnguard.data import preprocess_data
from churnguard.train import RandomForestClassifierParameters, train_model
from sklearn.pipeline import Pipeline


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    """Minimal dataset for training unit tests."""
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003"],
            "gender": ["Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "Yes", "No"],
            "tenure": [1, 12, 5],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No"],
            "OnlineBackup": ["Yes", "No", "Yes"],
            "DeviceProtection": ["No", "Yes", "No"],
            "TechSupport": ["No", "Yes", "No"],
            "StreamingTV": ["No", "Yes", "No"],
            "StreamingMovies": ["No", "Yes", "No"],
            "Contract": ["Month-to-month", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
            "MonthlyCharges": [29.85, 56.95, 70.00],
            "TotalCharges": ["29.85", "1889.5", "70.0"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_train_model_returns_fitted_pipeline(
    sample_training_data: pd.DataFrame,
) -> None:
    """Verify that training returns a fitted sklearn Pipeline."""
    features, target = preprocess_data(sample_training_data)

    model: Pipeline = train_model(
        features=features,
        target=target,
        model_name="random_forest",
        model_parameters=RandomForestClassifierParameters(),
    )

    assert isinstance(model, Pipeline)

    assert hasattr(model, "predict")
    assert hasattr(model, "fit")

    predictions = model.predict(features)

    assert isinstance(predictions, (list, pd.Series, np.ndarray))

    assert len(predictions) == len(target)

    values = np.asarray(predictions)
    unique_values = set(values.tolist())
    assert unique_values.issubset({0, 1})

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

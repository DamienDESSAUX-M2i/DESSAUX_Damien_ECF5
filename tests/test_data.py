from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from churnguard.data import load_data, preprocess_data


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a minimal valid dataframe for preprocessing tests."""
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 12],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": ["29.85", "1889.5"],
            "Churn": ["No", "Yes"],
        }
    )


def test_load_data_returns_dataframe(tmp_path: Path) -> None:
    """Verify that load_data returns a DataFrame."""
    file_path = tmp_path / "data.csv"

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(file_path, index=False)

    loaded = load_data(file_path)

    assert isinstance(loaded, pd.DataFrame)
    assert not loaded.empty


def test_load_data_has_expected_columns(tmp_path: Path) -> None:
    """Verify the presence of the expected 21 columns."""
    expected_columns = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    file_path = tmp_path / "telco.csv"
    df = pd.DataFrame({col: [None] for col in expected_columns})
    df.to_csv(file_path, index=False)

    loaded = load_data(file_path)

    assert set(loaded.columns) == expected_columns


def test_preprocess_returns_features_and_target(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Verify that preprocess_data splits features and target correctly."""
    features, target = preprocess_data(sample_dataframe)

    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)

    assert "Churn" not in features.columns
    assert target.name == "Churn" or target.name is None

    assert set(target.unique()).issubset({0, 1})
    assert len(features) == len(target)


def test_preprocess_handles_missing_total_charges() -> None:
    """Verify that empty TotalCharges values are handled correctly."""
    df = pd.DataFrame(
        {
            "customerID": ["0001", "0002"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 12],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [" ", "1889.5"],  # Invalid value
            "Churn": ["No", "Yes"],
        }
    )

    features, target = preprocess_data(df)

    # The line containing the invalid TotalCharges value must be deleted.
    assert len(features) == 1
    assert len(target) == 1

    # Verify that the conversion has taken place successfully.
    assert pd.api.types.is_numeric_dtype(features["TotalCharges"])

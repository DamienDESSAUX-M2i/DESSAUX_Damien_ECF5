from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data handling."""

    data_path: Path
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


DEFAULT_CONFIG = DataConfig(
    data_path=Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv",
)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""


def load_data(path: Path) -> pd.DataFrame:
    """Load dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataValidationError: If the file is empty or unreadable.

    """
    logger.info(f"Loading data from {path}")

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exception:
        raise DataValidationError(f"Failed to read CSV: {path}") from exception

    if df.empty:
        raise DataValidationError("Loaded dataset is empty")

    logger.info(f"Data loaded successfully with shape {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess the raw dataset.

    Steps:
        - Convert TotalCharges to numeric
        - Drop rows with missing values
        - Remove identifier columns
        - Split features and target

    Args:
        df: Raw input dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target.

    Raises:
        DataValidationError: If required columns are missing.

    """
    logger.info("Starting preprocessing")

    required_columns = {"TotalCharges", "Churn", "customerID"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    initial_shape = df.shape
    df = df.dropna()
    logger.info(f"Dropped NA values: {initial_shape[0] - df.shape[0]} rows removed")

    df = df.drop(columns=["customerID"])

    target = (df["Churn"] == "Yes").astype("int64")
    features = df.drop(columns=["Churn"])

    if features.empty:
        raise DataValidationError("Features dataframe is empty after preprocessing")

    logger.info(f"Preprocessing completed. Features shape: {features.shape}")

    return features, target


def split_data_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train and test sets.

    Args:
        features: Feature dataframe.
        target: Target series.
        config: Data configuration.

    Returns:
        Tuple containing train/test splits.

    Raises:
        DataValidationError: If input data is invalid.

    """
    logger.info(f"Splitting data with test_size={config.test_size}")

    if len(features) != len(target):
        raise DataValidationError("Features and target length mismatch")

    if features.empty or target.empty:
        raise DataValidationError("Features or target is empty")

    stratify = target if config.stratify else None

    try:
        features_train, features_test, target_train, target_test = train_test_split(
            features,
            target,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise DataValidationError("Train/test split failed") from exc

    logger.info(f"Split completed: train={features_train.shape}, test={features_test.shape}")

    return features_train, features_test, target_train, target_test

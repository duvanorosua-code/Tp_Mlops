"""Original data cleaning, feature engineering, sampling, and sklearn transformer."""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import BANDS, CLASS_MAPPING, COLOR_PAIRS, FEATURES, RAW_FEATURES, RANDOM_STATE, TEST_SIZE, VALIDATION_SAMPLE_SIZE

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """Stratified train/test data and class encoding."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    class_mapping: dict[str, int]
    sample_size: int


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with -9999 in any band and remove the constant rerun_ID column."""
    required = [*BANDS, "rerun_ID", "class"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns required for cleaning: {missing}")
    sentinel_mask = (data[BANDS] == -9999).any(axis=1)
    cleaned = data.loc[~sentinel_mask].drop(columns=["rerun_ID"]).reset_index(drop=True)
    LOGGER.info("Removed %d sentinel rows; %d rows remain", int(sentinel_mask.sum()), len(cleaned))
    return cleaned


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the four notebook colors and signed_log_redshift without changing formulas."""
    missing = [column for column in RAW_FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")
    featured = data.copy()
    for left, right in COLOR_PAIRS:
        featured[f"{left}_{right}"] = featured[left] - featured[right]
    featured["signed_log_redshift"] = (
        np.sign(featured["redshift"]) * np.log1p(np.abs(featured["redshift"]))
    )
    return featured


def encode_classes(labels: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Use the same alphabetical LabelEncoder mapping as the notebook."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    mapping = {str(name): int(index) for index, name in enumerate(encoder.classes_)}
    if mapping != CLASS_MAPPING:
        raise ValueError(f"Unexpected class mapping {mapping}; expected {CLASS_MAPPING}")
    return pd.Series(encoded, index=labels.index, name="class_encoded"), mapping


def prepare_sample(
    data: pd.DataFrame,
    sample_size: int = VALIDATION_SAMPLE_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DataSplit:
    """Take a reproducible stratified sample, then a stratified train/test split."""
    if "class" not in data.columns:
        raise ValueError("Target column 'class' is missing.")
    if sample_size <= 0 or sample_size > len(data):
        raise ValueError(f"sample_size must be between 1 and {len(data)}; got {sample_size}")
    sample, _ = train_test_split(
        data, train_size=sample_size, stratify=data["class"], random_state=random_state
    )
    y, mapping = encode_classes(sample["class"])
    X = sample.loc[:, RAW_FEATURES].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    expected = set(mapping.values())
    if set(y_train.unique()) != expected or set(y_test.unique()) != expected:
        raise ValueError("All three classes must be present in train and test.")
    LOGGER.info("Sample=%d train=%d test=%d class_mapping=%s",
                len(sample), len(X_train), len(X_test), mapping)
    return DataSplit(X_train, X_test, y_train, y_test, mapping, len(sample))


class AstronomyFeatureTransformer(BaseEstimator, TransformerMixin):
    """Reusable sklearn transformer so training and inference share identical features."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AstronomyFeatureTransformer":
        self._validate(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        if (X[BANDS] == -9999).any(axis=None):
            raise ValueError("Input contains -9999 sentinel values; clean or reject those rows first.")
        return create_features(X).loc[:, FEATURES].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(FEATURES, dtype=object)

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AstronomyFeatureTransformer requires a pandas DataFrame.")
        missing = [column for column in RAW_FEATURES if column not in X.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")

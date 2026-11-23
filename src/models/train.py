"""Build and train the XGBoost estimator."""
from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODEL_HYPERPARAMETERS
from src.features.preprocessing import AstronomyFeatureTransformer


def build_model(hyperparameters: Mapping[str, Any] | None = None) -> Pipeline:
    """Return feature-transformer + XGBoost pipeline.

    Explicit overrides are intended for tiny synthetic unit tests only.
    """
    parameters = dict(MODEL_HYPERPARAMETERS if hyperparameters is None else hyperparameters)
    return Pipeline([
        ("features", AstronomyFeatureTransformer()),
        ("model", XGBClassifier(**parameters)),
    ])


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Fit only XGBoost; no CV or hyperparameter search is performed."""
    model = build_model(hyperparameters)
    model.fit(X_train, y_train)
    return model

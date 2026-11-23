"""Synthetic fixtures; tests never call KaggleHub."""
import pandas as pd
import pytest

from src.config import CLASS_MAPPING, RAW_FEATURES
from src.models.train import train_model


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    rows = []
    for class_name, offset in (("GALAXY", 0.0), ("QSO", 2.0), ("STAR", -1.0)):
        for index in range(30):
            bands = ["u", "g", "r", "i", "z"]
            row = {name: offset + index * 0.001 + band_index * 0.1
                   for band_index, name in enumerate(bands)}
            row.update({
                "redshift": offset + index * 0.0001,
                "class": class_name,
                "rerun_ID": 301,
                "spec_obj_ID": f"{class_name}-{index}",
            })
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_trained_model():
    """Fit three trees once for fast serialization and prediction checks."""
    rows = []
    labels = []
    for class_id, offset in enumerate((0.0, 2.0, -1.0)):
        for index in range(12):
            row = {name: offset + index * 0.01 + band_index * 0.1
                   for band_index, name in enumerate(RAW_FEATURES[:-1])}
            row["redshift"] = offset + index * 0.001
            rows.append(row)
            labels.append(class_id)
    X = pd.DataFrame(rows, columns=RAW_FEATURES)
    y = pd.Series(labels)
    test_parameters = {
        "n_estimators": 3,
        "learning_rate": 0.1,
        "max_depth": 2,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_jobs": 1,
        "random_state": 42,
    }
    return train_model(X, y, hyperparameters=test_parameters), X

"""Reproducible configuration for the SDSS17 project."""
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "fedesoriano/stellar-classification-dataset-sdss17"
CSV_NAME = "star_classification.csv"
RANDOM_STATE = 42
VALIDATION_SAMPLE_SIZE = 5_000
TEST_SIZE = 0.20

BANDS = ["u", "g", "r", "i", "z"]
RAW_FEATURES = [*BANDS, "redshift"]
COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
COLOR_FEATURES = [f"{a}_{b}" for a, b in COLOR_PAIRS]
FEATURES = [*BANDS, *COLOR_FEATURES, "signed_log_redshift"]
# Same LabelEncoder alphabetical mapping as the original notebook.
CLASS_MAPPING = {"GALAXY": 0, "QSO": 1, "STAR": 2}

MODEL_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 587,
    "learning_rate": 0.0167,
    "max_depth": 9,
    "min_child_weight": 2,
    "subsample": 0.8253,
    "colsample_bytree": 0.8430,
    "reg_lambda": 5.8723,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

MLFLOW_EXPERIMENT_NAME = "sdss17-xgboost-validation"
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlartifacts"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
TRANSFORMATIONS = {
    "color_features": {f"{a}_{b}": f"{a} - {b}" for a, b in COLOR_PAIRS},
    "signed_log_redshift": "sign(redshift) * log1p(abs(redshift))",
    "sentinel_rule": "drop rows where any band equals -9999",
    "constant_column_rule": "drop rerun_ID",
    "dropped_source_column": "obj_ID (truncated in source dataset)",
}

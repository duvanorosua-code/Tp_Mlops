"""Local training pipeline orchestration and MLflow tracking."""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    CLASS_MAPPING, FEATURES, MLFLOW_ARTIFACTS, MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI, MODEL_HYPERPARAMETERS, PROJECT_ROOT, RANDOM_STATE,
    TEST_SIZE, TRANSFORMATIONS, VALIDATION_SAMPLE_SIZE,
)
from src.data.data_loader import load_data
from src.evaluation.evaluate import evaluate_model
from src.features.preprocessing import clean_data, prepare_sample
from src.models.artifacts import build_artifact_bundle, save_artifacts
from src.models.train import train_model

LOGGER = logging.getLogger(__name__)


def configure_mlflow() -> str:
    """Configure local SQLite tracking and a project-local artifact directory."""
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACTS.as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    LOGGER.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    LOGGER.info("MLflow artifact directory: %s", MLFLOW_ARTIFACTS)
    return MLFLOW_TRACKING_URI


def _log_active_run(
    results: dict[str, Any],
    metadata: dict[str, Any],
    model_path: Path,
    metadata_path: Path,
    results_path: Path,
) -> None:
    """Log run parameters, scores, confusion matrix, and persisted bundle."""
    params = {key: str(value) for key, value in MODEL_HYPERPARAMETERS.items()}
    params.update({
        "sample_size": str(metadata["sample_size"]),
        "train_size": str(metadata["train_size"]),
        "test_size": str(metadata["test_size"]),
        "features": json.dumps(FEATURES, ensure_ascii=False),
        "class_mapping": json.dumps(CLASS_MAPPING, ensure_ascii=False, sort_keys=True),
    })
    mlflow.log_params(params)
    mlflow.set_tags({
        "model": "XGBClassifier",
        "dataset": metadata["dataset_id"],
        "execution_type": metadata["execution_type"],
        "validation_only": "true",
    })
    mlflow.log_metrics({"accuracy": results["accuracy"], "f1_macro": results["f1_macro"]})
    mlflow.log_dict({
        "labels": results["confusion_matrix_labels"],
        "matrix": results["confusion_matrix"],
    }, "evaluation/confusion_matrix.json")
    mlflow.log_artifact(str(model_path), artifact_path="joblib_model")
    mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
    mlflow.log_artifact(str(results_path), artifact_path="evaluation")


def run_training_pipeline(sample_size: int = VALIDATION_SAMPLE_SIZE) -> dict[str, Any]:
    """Run one reduced validation; every invocation creates a fresh MLflow run."""
    configure_mlflow()
    run_name = (
        f"sdss17-xgb-validation-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    LOGGER.info("Starting training pipeline: %s", run_name)

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id
        raw_data = load_data()
        cleaned = clean_data(raw_data)
        split = prepare_sample(cleaned, sample_size=sample_size)

        LOGGER.info("Training XGBoost on %d rows with original selected parameters", len(split.X_train))
        model = train_model(split.X_train, split.y_train)
        results = evaluate_model(model, split.X_test, split.y_test)

        metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "execution_type": "technical_validation_sample",
            "dataset_id": "fedesoriano/stellar-classification-dataset-sdss17",
            "sample_size": split.sample_size,
            "train_size": len(split.X_train),
            "test_size": len(split.X_test),
            "test_fraction": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "features": list(FEATURES),
            "class_mapping": dict(CLASS_MAPPING),
            "hyperparameters": dict(MODEL_HYPERPARAMETERS),
            "transformations": TRANSFORMATIONS,
            "metrics_note": (
                "Reduced technical validation; metrics are not directly comparable "
                "with the original notebook evaluation."
            ),
        }
        bundle = build_artifact_bundle(model, metadata)
        models_dir = PROJECT_ROOT / "models"
        outputs_dir = PROJECT_ROOT / "outputs"
        model_path = models_dir / f"xgboost_{run_id}.joblib"
        metadata_path = models_dir / f"xgboost_{run_id}_metadata.json"
        results_path = outputs_dir / f"validation_{run_id}.json"
        save_artifacts(bundle, model_path, metadata_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps({**results, "metadata": metadata}, indent=2,
                       ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _log_active_run(results, metadata, model_path, metadata_path, results_path)
        LOGGER.info("Saved model and evaluation artifacts for run %s", run_id)

    final = {
        **results,
        "run_id": run_id,
        "run_name": run_name,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "results_path": str(results_path),
        "tracking_uri": MLFLOW_TRACKING_URI,
    }
    LOGGER.info("Pipeline completed successfully; MLflow run_id=%s", run_id)
    return final


def get_mlflow_client() -> MlflowClient:
    """Return a client configured to query the local tracking store."""
    configure_mlflow()
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

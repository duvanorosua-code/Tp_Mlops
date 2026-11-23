"""Versioned model bundles with feature and transform metadata."""
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from src.config import CLASS_MAPPING, FEATURES, MODEL_HYPERPARAMETERS, RAW_FEATURES, TRANSFORMATIONS


def build_artifact_bundle(model: Pipeline, metadata: dict[str, Any]) -> dict[str, Any]:
    """Bundle estimator, schema, class map, hyperparameters, and transform contract."""
    return {
        "model": model,
        "features": list(FEATURES),
        "raw_features": list(RAW_FEATURES),
        "class_mapping": dict(CLASS_MAPPING),
        "hyperparameters": dict(MODEL_HYPERPARAMETERS),
        "transformations": TRANSFORMATIONS,
        "metadata": metadata,
    }


def save_artifacts(
    bundle: dict[str, Any], model_path: Path, metadata_path: Path
) -> tuple[Path, Path]:
    """Save new files only; refuse to overwrite any existing artifact."""
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError("Artifact destination already exists; refusing to overwrite.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    public_metadata = {key: value for key, value in bundle.items() if key != "model"}
    metadata_path.write_text(
        json.dumps(public_metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return model_path, metadata_path


def load_artifact_bundle(path: str | Path) -> dict[str, Any]:
    """Load a trusted joblib bundle and check schema compatibility."""
    bundle = joblib.load(Path(path))
    required = {
        "model", "features", "raw_features", "class_mapping",
        "hyperparameters", "transformations", "metadata",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Artifact is missing required fields: {sorted(missing)}")
    if bundle["features"] != FEATURES or bundle["class_mapping"] != CLASS_MAPPING:
        raise ValueError("Artifact features or class mapping differ from project configuration.")
    return bundle

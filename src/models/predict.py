"""Load a model bundle and predict new SDSS observations."""
import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_FEATURES
from src.models.artifacts import load_artifact_bundle

LOGGER = logging.getLogger(__name__)


def predict_records(records: pd.DataFrame, artifact_path: str | Path) -> pd.DataFrame:
    """Predict class IDs and labels; packaged transformer computes training features."""
    missing = [column for column in RAW_FEATURES if column not in records.columns]
    if missing:
        raise ValueError(f"Missing raw input columns: {missing}")
    bundle = load_artifact_bundle(artifact_path)
    raw = records.loc[:, RAW_FEATURES].copy()
    encoded = bundle["model"].predict(raw)
    reverse_mapping = {value: key for key, value in bundle["class_mapping"].items()}
    labels = [reverse_mapping[int(value)] for value in encoded]
    LOGGER.info("Generated predictions for %d records", len(records))
    return pd.DataFrame(
        {"class_id": encoded.astype(int), "predicted_class": labels},
        index=records.index,
    )

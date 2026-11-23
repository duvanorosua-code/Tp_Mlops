"""Holdout metrics and confusion matrix."""
import logging
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src.config import CLASS_MAPPING

LOGGER = logging.getLogger(__name__)


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    """Evaluate holdout only and return JSON-compatible metrics."""
    names = list(CLASS_MAPPING)
    ids = [CLASS_MAPPING[name] for name in names]
    predictions = model.predict(X_test)
    report = classification_report(
        y_test, predictions, labels=ids, target_names=names,
        output_dict=True, zero_division=0,
    )
    results = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=ids).tolist(),
        "confusion_matrix_labels": names,
    }
    readable_report = classification_report(
        y_test, predictions, labels=ids, target_names=names, digits=4, zero_division=0
    )
    readable_matrix = pd.DataFrame(results["confusion_matrix"], index=names, columns=names)
    LOGGER.info("Holdout accuracy=%.6f; F1 macro=%.6f", results["accuracy"], results["f1_macro"])
    LOGGER.info("Classification report:\n%s", readable_report)
    LOGGER.info("Confusion matrix (rows=true, columns=predicted):\n%s", readable_matrix)
    return results

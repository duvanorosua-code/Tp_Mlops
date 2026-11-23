"""Load SDSS17 data through the KaggleHub cache."""
import logging
from pathlib import Path

import kagglehub
import pandas as pd

from src.config import CSV_NAME, DATASET_ID

LOGGER = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Load the CSV from KaggleHub's cache, downloading only when needed."""
    dataset_dir = Path(kagglehub.dataset_download(DATASET_ID))
    csv_path = dataset_dir / CSV_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected {CSV_NAME} inside KaggleHub dataset path {dataset_dir}")
    LOGGER.info("Loading cached KaggleHub dataset %s", DATASET_ID)
    data = pd.read_csv(csv_path, dtype={"obj_ID": str, "spec_obj_ID": str})
    # The original notebook drops obj_ID because its values were truncated.
    if "obj_ID" in data.columns:
        data = data.drop(columns=["obj_ID"])
    LOGGER.info("Loaded %s rows and %s columns", f"{len(data):,}", data.shape[1])
    return data

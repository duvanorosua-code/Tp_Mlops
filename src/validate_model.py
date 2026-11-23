"""CLI entry point for the reduced technical validation."""
import argparse
import logging

from src.config import VALIDATION_SAMPLE_SIZE
from src.pipelines.training_pipeline import run_training_pipeline


def main() -> None:
    """Run the reproducible reduced validation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=VALIDATION_SAMPLE_SIZE,
        help=f"Stratified sample size (default: {VALIDATION_SAMPLE_SIZE}).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    try:
        run_training_pipeline(sample_size=args.sample_size)
    except Exception:
        logging.getLogger(__name__).exception("Validation pipeline failed")
        raise


if __name__ == "__main__":
    main()

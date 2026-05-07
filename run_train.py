from __future__ import annotations

import argparse
import logging

from churnguard.pipeline import run
from churnguard.train import (
    GradientBoostingClassifierParameters,
    LogisticRegressionParameters,
    ModelName,
    ModelParameters,
    RandomForestClassifierParameters,
)

logger = logging.getLogger(__name__)


MODEL_MAPPING: dict[str, tuple[ModelName, ModelParameters]] = {
    "lr": (ModelName.LOGISTIC_REGRESSION, LogisticRegressionParameters()),
    "rf": (ModelName.RANDOM_FOREST, RandomForestClassifierParameters()),
    "gb": (ModelName.GRADIENT_BOOSTING, GradientBoostingClassifierParameters()),
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.

    """
    parser = argparse.ArgumentParser(
        description="Run churn prediction pipeline for one or all models."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[*MODEL_MAPPING.keys(), "all"],
        help=(
            "Model to run: lr (LogisticRegression), rf (RandomForest), "
            "gb (GradientBoosting), or all."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Execute the command line interface entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    model_key: str = args.model

    if model_key == "all":
        logger.info("Run pipeline for all models")

        for key, (model_name, model_parameters) in MODEL_MAPPING.items():
            logger.info(f"Start run for model={key}")
            run(model_name=model_name, model_parameters=model_parameters)

        logger.info("All model runs completed")
        return

    model_name, model_parameters = MODEL_MAPPING[model_key]

    logger.info(f"Selected model: {model_key}")

    run(model_name=model_name, model_parameters=model_parameters)

    logger.info("Pipeline execution completed")


if __name__ == "__main__":
    main()

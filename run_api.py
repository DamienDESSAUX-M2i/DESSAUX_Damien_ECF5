import logging

import uvicorn

logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the command line interface entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Run API")
    uvicorn.run(app="api.main:app", host="0.0.0.0", port=8000)
    logger.info("API shut down")


if __name__ == "__main__":
    main()

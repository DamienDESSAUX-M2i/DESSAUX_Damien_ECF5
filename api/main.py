from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, cast

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request

from .model_loader import ModelLoader
from .schemas import HealthResponse, PredictionRequest, PredictionResponse
from .settings import MLFLOW_SETTINGS

logger = logging.getLogger(__name__)


def to_churn_label(p: int) -> str:
    """Convert prediction class to churn label."""
    return "true" if p == 1 else "false"


def get_model_loader(request: Request) -> ModelLoader:
    """Retrieve model loader from application state."""
    return cast(ModelLoader, request.app.state.model_loader)


ModelDep = Annotated[ModelLoader, Depends(get_model_loader)]


def predict_internal(
    model: ModelLoader,
    df: pd.DataFrame,
) -> tuple[list[str], list[float | None]]:
    """Run prediction and return churn label and probabilities."""
    preds = model.predict(df)
    churns = [to_churn_label(p) for p in preds]

    probabilities: list[float | None] = []
    try:
        probas = model.predict_proba(df)
        probabilities = [row[1] for row in probas]
    except NotImplementedError:
        logger.warning("predict_proba not available")
        probabilities = [None for p in preds]

    return churns, probabilities


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, ModelLoader]:
    """Create lifespan for FastAPI."""
    logger.info("Loading model")

    app.state.model_loader = ModelLoader()

    if not app.state.model_loader.is_loaded:
        logger.error("Model failed to load")
    else:
        logger.info("Model loaded successfully")

    yield

    logger.info("Shutting down API")


def create_app() -> FastAPI:
    """Create FastAPI application with lifecycle management."""
    app = FastAPI(
        title="Churn Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(model: ModelDep) -> HealthResponse:
        """Return service health status."""
        status = "healthy" if model.is_loaded else "unhealthy"

        return HealthResponse(
            status=status,
            model=MLFLOW_SETTINGS.registered_model_name,
            version=model.version,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        request: PredictionRequest,
        model: ModelDep,
    ) -> PredictionResponse:
        """Run prediction on a single batch."""
        if not model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            df = pd.DataFrame([r.model_dump() for r in request.instances])

            churns, probabilities = predict_internal(model, df)

            return PredictionResponse(
                churns=churns,
                probabilities=probabilities,
            )

        except Exception as exception:
            logger.error(f"Prediction error: {exception}")
            raise HTTPException(status_code=500, detail="Prediction failed") from exception

    @app.post("/predict/batch", response_model=PredictionResponse)
    def predict_batch(
        request: PredictionRequest,
        model: ModelDep,
    ) -> PredictionResponse:
        """Run batch prediction with size validation."""
        if not model.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if len(request.instances) > 100:
            raise HTTPException(status_code=400, detail="Max batch size is 100")

        try:
            df = pd.DataFrame([r.model_dump() for r in request.instances])

            churns, probabilities = predict_internal(model, df)

            return PredictionResponse(
                churns=churns,
                probabilities=probabilities,
            )

        except Exception as exception:
            logger.error(f"Batch prediction error: {exception}")
            raise HTTPException(status_code=500, detail="Batch prediction failed") from exception

    return app


app = create_app()

"""
Serve MLflow models via FastAPI with health check and prediction endpoints.

This script demonstrates:
- Loading a model from the MLflow registry on startup
- Health check endpoint for load balancers and orchestrators
- Prediction endpoint accepting JSON payloads
- Batch prediction endpoint for multiple samples
- Model metadata endpoint exposing version and stage info

Usage:
    uvicorn src.serving.serve_model:app --host 0.0.0.0 --port 8000 --reload
    python -m src.serving.serve_model

Environment variables:
    MLFLOW_TRACKING_URI  - MLflow tracking server URL (default: http://localhost:5000)
    MODEL_NAME           - Registered model name (default: sklearn-wine-classifier)
    MODEL_STAGE          - Model stage to load (default: Production)
    SERVE_HOST           - Host to bind to (default: 0.0.0.0)
    SERVE_PORT           - Port to bind to (default: 8000)
"""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---- Configuration ----

class AppConfig:
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MODEL_NAME = os.getenv("MODEL_NAME", "sklearn-wine-classifier")
    MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
    SERVE_HOST = os.getenv("SERVE_HOST", "0.0.0.0")
    SERVE_PORT = int(os.getenv("SERVE_PORT", "8000"))


# ---- Request/Response Schemas ----

class PredictionRequest(BaseModel):
    """Single prediction request with feature values."""
    features: List[float] = Field(
        ...,
        description="List of feature values in the same order as training data",
        min_length=1,
    )
    feature_names: Optional[List[str]] = Field(
        None,
        description="Optional feature names for validation",
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "features": [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0],
            "feature_names": None,
        }]
    }}


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple samples."""
    instances: List[List[float]] = Field(
        ...,
        description="List of feature vectors",
        min_length=1,
    )
    feature_names: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    """Single prediction response."""
    prediction: Any
    model_name: str
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[Any]
    count: int
    model_name: str
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str]
    model_version: Optional[str]
    uptime_seconds: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str
    model_stage: str
    model_version: str
    tracking_uri: str
    loaded_at: str
    input_schema: Optional[Dict] = None


# ---- Application State ----

class ModelState:
    """Holds the loaded model and metadata."""
    model = None
    model_name: str = ""
    model_version: str = "unknown"
    model_stage: str = ""
    loaded_at: str = ""
    start_time: float = 0
    input_schema = None


state = ModelState()


def load_model():
    """Load model from MLflow registry."""
    config = AppConfig()
    mlflow.set_tracking_uri(config.TRACKING_URI)

    model_uri = f"models:/{config.MODEL_NAME}/{config.MODEL_STAGE}"
    print(f"Loading model from: {model_uri}")

    try:
        state.model = mlflow.pyfunc.load_model(model_uri)
        state.model_name = config.MODEL_NAME
        state.model_stage = config.MODEL_STAGE
        state.loaded_at = datetime.utcnow().isoformat()
        state.start_time = time.time()

        # Try to extract version info
        try:
            model_info = state.model.metadata
            state.model_version = str(getattr(model_info, "run_id", "unknown")[:8])
            if hasattr(model_info, "signature") and model_info.signature:
                state.input_schema = model_info.signature.inputs.to_dict() if model_info.signature.inputs else None
        except Exception:
            pass

        print(f"Model loaded successfully: {config.MODEL_NAME} ({config.MODEL_STAGE})")

    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
        print("Server will start without a model. Load one via the registry.")
        state.model = None


# ---- FastAPI App ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield
    print("Shutting down model server.")


app = FastAPI(
    title="MLflow Model Serving API",
    description="Serve MLflow registered models with FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for load balancers and orchestrators.
    Returns 200 if the server is running, regardless of model state.
    """
    uptime = time.time() - state.start_time if state.start_time else 0

    return HealthResponse(
        status="healthy" if state.model else "degraded",
        model_loaded=state.model is not None,
        model_name=state.model_name or None,
        model_version=state.model_version or None,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return metadata about the currently loaded model."""
    if not state.model:
        raise HTTPException(status_code=503, detail="No model loaded")

    return ModelInfoResponse(
        model_name=state.model_name,
        model_stage=state.model_stage,
        model_version=state.model_version,
        tracking_uri=AppConfig.TRACKING_URI,
        loaded_at=state.loaded_at,
        input_schema=state.input_schema,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a single prediction using the loaded model.

    Send a JSON body with a `features` array containing the feature values
    in the same order as the training data.
    """
    if not state.model:
        raise HTTPException(status_code=503, detail="No model loaded. Check /health.")

    try:
        # Build input DataFrame
        if request.feature_names:
            input_df = pd.DataFrame([request.features], columns=request.feature_names)
        else:
            input_df = pd.DataFrame([request.features])

        # Make prediction
        prediction = state.model.predict(input_df)

        # Convert numpy types to Python native
        pred_value = prediction[0]
        if isinstance(pred_value, (np.integer,)):
            pred_value = int(pred_value)
        elif isinstance(pred_value, (np.floating,)):
            pred_value = float(pred_value)

        return PredictionResponse(
            prediction=pred_value,
            model_name=state.model_name,
            model_version=state.model_version,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple samples.

    Send a JSON body with an `instances` array where each element is
    a feature vector.
    """
    if not state.model:
        raise HTTPException(status_code=503, detail="No model loaded. Check /health.")

    try:
        if request.feature_names:
            input_df = pd.DataFrame(request.instances, columns=request.feature_names)
        else:
            input_df = pd.DataFrame(request.instances)

        predictions = state.model.predict(input_df)

        # Convert numpy array to Python list
        pred_list = []
        for p in predictions:
            if isinstance(p, (np.integer,)):
                pred_list.append(int(p))
            elif isinstance(p, (np.floating,)):
                pred_list.append(float(p))
            else:
                pred_list.append(p)

        return BatchPredictionResponse(
            predictions=pred_list,
            count=len(pred_list),
            model_name=state.model_name,
            model_version=state.model_version,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload the model from the registry (useful after promoting a new version)."""
    try:
        load_model()
        if state.model:
            return {"status": "reloaded", "model_name": state.model_name}
        else:
            raise HTTPException(status_code=503, detail="Model reload failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload error: {str(e)}")


def main():
    """Run the serving application."""
    config = AppConfig()
    print(f"Starting MLflow Model Server on {config.SERVE_HOST}:{config.SERVE_PORT}")
    print(f"Model: {config.MODEL_NAME} ({config.MODEL_STAGE})")
    print(f"Tracking URI: {config.TRACKING_URI}")

    uvicorn.run(
        "src.serving.serve_model:app",
        host=config.SERVE_HOST,
        port=config.SERVE_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

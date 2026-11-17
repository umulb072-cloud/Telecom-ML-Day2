
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

from backend.model_service import model_service, FEATURE_NAMES


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_model()
    print("Telecom AI model loaded successfully!")
    yield


app = FastAPI(
    title="Telecom AI Prediction API",
    description="Telecom Network Anomaly Detection",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "message": "Welcome to Telecom AI API"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "model_name": model_service.model_name
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = [
            getattr(request, name)
            for name in FEATURE_NAMES
        ]

        prediction, probability = model_service.predict(features)

        return {
            "request_id": request.request_id,
            "prediction": prediction,
            "anomaly_probability": round(probability, 4),
            "message": "Prediction generated successfully"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

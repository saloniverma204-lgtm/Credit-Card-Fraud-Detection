
import joblib
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response


# 1. CREATE FASTAPI APPLICATION

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-based fraud detection service",
    version="1.0"
)


# 2. LOAD TRAINED MODEL SAFELY

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

model = None


@app.on_event("startup")
def load_trained_model():
    global model

    try:
        model = joblib.load(MODEL_PATH)
        print("ML model loaded successfully")

    except FileNotFoundError:
        model = None
        print("Model file not found. API started without model.")


# 3. EXPECTED INPUT FEATURES

FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


# 4. PROMETHEUS METRICS

prediction_requests = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["result"]
)


# 5. INPUT VALIDATION

class Transaction(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Transaction features including Time, V1-V28 and Amount"
    )


# 6. HOME ROUTE

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "status": "success"
    }


# 7. HEALTH CHECK

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# 8. FRAUD PREDICTION

@app.post("/predict")
def predict(transaction: Transaction):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not available"
        )

    features = transaction.features

    missing = set(FEATURE_COLUMNS) - set(features.keys())
    extra = set(features.keys()) - set(FEATURE_COLUMNS)

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_features": sorted(missing),
                "extra_features": sorted(extra)
            }
        )

    input_data = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    prediction = int(model.predict(input_data)[0])

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    result = "fraud" if prediction == 1 else "genuine"

    prediction_requests.labels(result=result).inc()

    return {
        "prediction": prediction,
        "result": result,
        "fraud_probability": round(probability, 4),
        "message": (
            "Warning: Potential fraudulent transaction"
            if prediction == 1
            else "Transaction classified as genuine"
        )
    }


# 9. PROMETHEUS METRICS ENDPOINT

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


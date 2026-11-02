
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# =====================================
# 1. CREATE FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-based fraud detection service",
    version="1.0"
)

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================

model = joblib.load("models/model.joblib")

# Expected input columns
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

# =====================================
# 3. PROMETHEUS METRICS
# =====================================

prediction_requests = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["result"]
)

# =====================================
# 4. INPUT DATA VALIDATION
# =====================================

class Transaction(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Transaction features including Time, V1-V28 and Amount"
    )

# =====================================
# 5. HOME ROUTE
# =====================================

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "status": "success"
    }

# =====================================
# 6. HEALTH CHECK
# =====================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

# =====================================
# 7. FRAUD PREDICTION
# =====================================

@app.post("/predict")
def predict(transaction: Transaction):

    features = transaction.features

    # Check whether all required features are present
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

    # Arrange input features in correct order
    input_data = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    # Make prediction
    prediction = int(model.predict(input_data)[0])

    # Calculate fraud probability
    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    # Update Prometheus counter
    result = "fraud" if prediction == 1 else "genuine"
    prediction_requests.labels(result=result).inc()

    # Return result
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

# =====================================
# 8. PROMETHEUS METRICS ENDPOINT
# =====================================

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

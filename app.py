import numpy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import time
from prometheus_client import Counter, Histogram, make_asgi_app


# Load model
model = joblib.load("model.pkl")

# Define Prometheus metrics
PRED_COUNTER = Counter(
    "iris_prediction_requests_total",
    "Total number of iris prediction requests"
)
PRED_LATENCY = Histogram(
    "iris_prediction_latency_seconds",
    "Latency for iris predictions"
)

# Define request body
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create FastAPI app
app = FastAPI(title="Iris Classifier")

# Mount Prometheus metrics endpoint at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.post("/predict")
async def predict(features: IrisFeatures):
    PRED_COUNTER.inc()
    start = time.time()

    data = pd.DataFrame([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]], columns=[
        "sepal length (cm)", "sepal width (cm)",
        "petal length (cm)", "petal width (cm)"
    ])

    try:
        pred = model.predict(data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    PRED_LATENCY.observe(time.time() - start)
    return {"predicted_class": int(pred)}

@app.get("/")
async def root():
    return {"message": "Iris classifier up. POST to /predict"}

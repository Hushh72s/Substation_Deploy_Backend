"""
api.py
------
Stateless FastAPI endpoint for substation anomaly detection.
Receives one sensor reading, runs IsolationForest, returns severity.

Deploy on Render:
  Start command: uvicorn api:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Substation Predict API",
    description="ML anomaly detection for substation sensor data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model on startup ─────────────────────────────────────────
model = None
try:
    model = joblib.load("model.pkl")
    print("✅  Model loaded from model.pkl")
except Exception as e:
    print(f"⚠  Could not load model.pkl: {e}")


# ── Request schema ────────────────────────────────────────────────
class SensorData(BaseModel):
    temperature: float
    humidity: float
    vibration: float
    voltage: float


# ── Severity classification (mirrors consumer_engine.py) ─────────
THRESHOLDS = {
    "temperature": {"warning": 85,  "critical": 95},
    "vibration":   {"warning": 3.5, "critical": 4.5},
    "voltage_low": {"warning": 208, "critical": 205},
    "voltage_high":{"warning": 242, "critical": 246},
    "humidity":    {"warning": 80,  "critical": 88},
}

def classify_severity(d: dict, is_anomaly: bool) -> str:
    temp  = d["temperature"]
    vib   = d["vibration"]
    volt  = d["voltage"]
    humid = d["humidity"]

    if (temp  >= THRESHOLDS["temperature"]["critical"]  or
        vib   >= THRESHOLDS["vibration"]["critical"]    or
        volt  <= THRESHOLDS["voltage_low"]["critical"]  or
        volt  >= THRESHOLDS["voltage_high"]["critical"] or
        humid >= THRESHOLDS["humidity"]["critical"]):
        return "CRITICAL"

    if (temp  >= THRESHOLDS["temperature"]["warning"]  or
        vib   >= THRESHOLDS["vibration"]["warning"]    or
        volt  <= THRESHOLDS["voltage_low"]["warning"]  or
        volt  >= THRESHOLDS["voltage_high"]["warning"] or
        humid >= THRESHOLDS["humidity"]["warning"]):
        return "WARNING"

    if is_anomaly:
        return "WARNING"

    return "NORMAL"


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — used by Render to verify the service is up."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: SensorData):
    """
    Run the IsolationForest model on one sensor reading.
    Returns anomaly flag, anomaly score, and severity label.
    """
    if model is None:
        return {"error": "Model not loaded on server.", "severity": "UNKNOWN"}

    d = data.dict()
    X = [[d["temperature"], d["humidity"], d["vibration"], d["voltage"]]]

    pred     = model.predict(X)[0]
    score    = round(float(model.score_samples(X)[0]), 4)
    anomaly  = bool(pred == -1)
    severity = classify_severity(d, anomaly)

    return {
        "anomaly":  anomaly,
        "score":    score,
        "severity": severity,
    }

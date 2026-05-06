import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Threshold constants MUST match training (Model/Main.py)
COOLANT_WARN = 108
COOLANT_CRIT = 115

OIL_WARN_LOAD = 15
OIL_CRIT = 10

MAP_WARN_KPA = 27.1
MAP_CRIT_KPA = 101.3

RPM_REDLINE = 6500


def _predict_status(
    model: Any,
    le: Any,
    features: list,
    coolant: float,
    oil: float,
    map_kpa: float,
    rpm: float,
) -> Tuple[str, Dict[str, float]]:
    row = pd.DataFrame(
        [
            {
                "coolant_c": coolant,
                "oil_psi": oil,
                "map_kpa": map_kpa,
                "rpm": rpm,
                "coolant_over_warn": max(0.0, coolant - COOLANT_WARN),
                "coolant_over_crit": max(0.0, coolant - COOLANT_CRIT),
                "oil_deficit": max(0.0, OIL_WARN_LOAD - oil),
                "oil_danger": max(0.0, OIL_CRIT - oil),
                "map_vacuum_deficit": max(0.0, MAP_WARN_KPA - map_kpa),
                "map_boost_excess": max(0.0, map_kpa - MAP_CRIT_KPA),
                "rpm_over_redline": max(0.0, rpm - RPM_REDLINE),
                "under_load": int(rpm > 1000),
                "oil_load_risk": max(0.0, OIL_WARN_LOAD - oil) * int(rpm > 1000),
            }
        ]
    )

    pred = model.predict(row[features])[0]
    proba = model.predict_proba(row[features])[0]

    label = le.inverse_transform([pred])[0]
    conf = {str(le.classes_[i]): float(proba[i]) for i in range(len(proba))}
    return str(label), conf


class TelemetryIn(BaseModel):
    coolant_c: float = Field(..., description="Coolant temperature in °C")
    oil_psi: float = Field(..., description="Oil pressure in PSI")
    map_kpa: float = Field(..., description="MAP pressure in kPa")
    rpm: float = Field(..., description="Engine RPM")


@dataclass
class EngineSnapshot:
    coolant_c: Optional[float] = None
    oil_psi: Optional[float] = None
    map_kpa: Optional[float] = None
    rpm: Optional[float] = None
    last_update_unix: Optional[float] = None
    last_status: Optional[str] = None
    last_confidence: Optional[Dict[str, float]] = None
    last_predict_unix: Optional[float] = None


class EngineService:
    def __init__(self) -> None:
        model_dir = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "Model"))

        self.model = joblib.load(os.path.join(model_dir, "engine_health_model.pkl"))
        self.le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        self.features = joblib.load(os.path.join(model_dir, "features.pkl"))

        self.state = EngineSnapshot()
        self._lock = threading.Lock()

    def predict(self, t: TelemetryIn) -> Dict[str, Any]:
        label, conf = _predict_status(
            self.model,
            self.le,
            self.features,
            coolant=float(t.coolant_c),
            oil=float(t.oil_psi),
            map_kpa=float(t.map_kpa),
            rpm=float(t.rpm),
        )
        return {
            "inputs": {
                "coolant_c": float(t.coolant_c),
                "oil_psi": float(t.oil_psi),
                "map_kpa": float(t.map_kpa),
                "rpm": float(t.rpm),
            },
            "prediction": {"status": label, "confidence": conf},
        }

    def update_latest(self, t: TelemetryIn) -> Dict[str, Any]:
        now = time.time()
        result = self.predict(t)
        with self._lock:
            self.state.coolant_c = float(t.coolant_c)
            self.state.oil_psi = float(t.oil_psi)
            self.state.map_kpa = float(t.map_kpa)
            self.state.rpm = float(t.rpm)
            self.state.last_update_unix = now
            self.state.last_status = str(result["prediction"]["status"])
            self.state.last_confidence = dict(result["prediction"]["confidence"])
            self.state.last_predict_unix = now
        return {
            **result,
            "last_update_unix": now,
            "predicted_at_unix": now,
        }

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "inputs": {
                    "coolant_c": self.state.coolant_c,
                    "oil_psi": self.state.oil_psi,
                    "map_kpa": self.state.map_kpa,
                    "rpm": self.state.rpm,
                },
                "last_update_unix": self.state.last_update_unix,
                "prediction": {
                    "status": self.state.last_status,
                    "confidence": self.state.last_confidence,
                    "predicted_at_unix": self.state.last_predict_unix,
                },
            }


app = FastAPI(title="Engine Health Predictor", version="1.0.0")
svc = EngineService()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"ok": "true"}


@app.post("/predict")
def predict(t: TelemetryIn) -> Dict[str, Any]:
    return svc.predict(t)


@app.post("/telemetry")
def telemetry(t: TelemetryIn) -> Dict[str, Any]:
    return svc.update_latest(t)


@app.get("/status")
def status() -> Dict[str, Any]:
    state = svc.get_state()
    inputs = state["inputs"]
    if any(inputs[k] is None for k in ("coolant_c", "oil_psi", "map_kpa", "rpm")):
        raise HTTPException(status_code=503, detail={"error": "no_telemetry_yet", "state": state})
    if state["prediction"]["status"] is None:
        raise HTTPException(status_code=503, detail={"error": "no_prediction_yet", "state": state})
    return state


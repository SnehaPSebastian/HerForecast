from pydantic import BaseModel
from typing import Dict, Optional

class WearableData(BaseModel):
    spo2: float
    gsr_mean: float
    gsr_phasic_std: float
    ppg_rmssd: float
    heart_rate: float
    skin_temp: float

class HormoneData(BaseModel):
    estrogen: float
    progesterone: float

class PredictRequest(BaseModel):
    wearable_data: WearableData
    hormone_data: HormoneData
    day_in_cycle: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class PredictResponse(BaseModel):
    predicted_phase: str
    predicted_mood: str
    confidence: float
    probabilities: Dict[str, float]

class ModelInfoResponse(BaseModel):
    model_name: str
    expected_features: list[str]

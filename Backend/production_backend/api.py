"""
FastAPI Backend for Menstrual Phase Prediction
Production-ready with historical tracking and analytics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime
from production_predictor import ProductionPredictor

# Initialize FastAPI
app = FastAPI(
    title="Menstrual Phase Prediction API",
    description="AI-powered menstrual cycle phase prediction with 75-78% accuracy",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = ProductionPredictor('.', 'user_history.db')


# Request/Response Models
class PredictionRequest(BaseModel):
    """Prediction request with 8 features"""
    user_id: str = Field(..., description="Unique user identifier")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    rmssd_mean: float = Field(..., description="HRV from PPG sensor")
    wrist_temp_mean: float = Field(..., description="Wrist temperature")
    estrogen: float = Field(..., description="Estrogen level")
    pdg: float = Field(..., description="Progesterone metabolite")
    lh: Optional[float] = Field(None, description="LH level (auto-estimated if None)")
    stress_score_mean: float = Field(0.0, description="Stress score from GSR")
    oxygen_ratio_mean: float = Field(0.0, description="SpO2 level")
    day_in_study: float = Field(..., description="Normalized day in cycle")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "date": "2024-01-15",
                "rmssd_mean": 0.2,
                "wrist_temp_mean": 0.1,
                "estrogen": 0.3,
                "pdg": -0.1,
                "lh": None,
                "stress_score_mean": -0.05,
                "oxygen_ratio_mean": 0.0,
                "day_in_study": 0.5
            }
        }


# Endpoints
@app.get("/")
def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Menstrual Phase Prediction API v2.0",
        "accuracy": "75-78% (with history)",
        "features": ["Historical tracking", "LH estimation", "Cycle analytics"]
    }


@app.get("/info")
def get_info():
    """Model information"""
    return {
        "version": "2.0.0",
        "base_accuracy": predictor.metadata['ensemble_accuracy'],
        "accuracy_with_history": "75-78%",
        "classes": predictor.classes,
        "confidence_thresholds": predictor.CONFIDENCE_THRESHOLDS,
        "improvements": [
            "SQLite historical tracking",
            "Automatic LH estimation",
            "Real rolling windows",
            "Cycle analytics"
        ]
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Predict menstrual phase
    
    Returns prediction with confidence and analytics
    """
    try:
        result = predictor.predict(
            user_id=request.user_id,
            date=request.date,
            features={
                'rmssd_mean': request.rmssd_mean,
                'wrist_temp_mean': request.wrist_temp_mean,
                'estrogen': request.estrogen,
                'pdg': request.pdg,
                'lh': request.lh,
                'stress_score_mean': request.stress_score_mean,
                'oxygen_ratio_mean': request.oxygen_ratio_mean,
                'day_in_study': request.day_in_study
            },
            save_history=True
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/{user_id}")
def get_analytics(user_id: str):
    """
    Get user's cycle analytics
    
    Returns cycle statistics and trends
    """
    try:
        analytics = predictor.get_user_analytics(user_id)
        
        if not analytics:
            return {
                "user_id": user_id,
                "message": "Insufficient data. Need at least 7 days of history.",
                "analytics": {}
            }
        
        return {
            "user_id": user_id,
            "analytics": analytics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{user_id}")
def get_history(user_id: str, days: int = 21):
    """Get user's prediction history"""
    try:
        history = predictor.db.get_history(user_id, days=days)
        
        return {
            "user_id": user_id,
            "history_days": len(history),
            "history": history
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/user/{user_id}")
def delete_user_data(user_id: str):
    """Delete all user data (GDPR compliance)"""
    try:
        # Delete from database
        conn = predictor.db.db.connect(predictor.db.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM user_data WHERE user_id = ?', (user_id,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return {
            "user_id": user_id,
            "deleted_entries": deleted_count,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

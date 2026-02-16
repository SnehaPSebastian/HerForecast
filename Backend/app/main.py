from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .model_loader import load_model, get_pipeline, get_feature_columns, get_label_classes
from .schemas import PredictRequest, PredictResponse, HealthResponse, ModelInfoResponse
from .prediction_service import preprocess_data, predict_phase
import traceback

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    load_model()
    yield

app = FastAPI(
    title="Inner Weather Backend",
    description="API for menstrual phase prediction using wearable data",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the API and model are loaded.
    """
    try:
        model_loaded = get_pipeline() is not None
        return HealthResponse(status="ok", model_loaded=model_loaded)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict menstrual phase and mood based on provided wearable and hormone data.
    """
    try:
        wearable_data = request.wearable_data.dict()
        hormone_data = request.hormone_data.dict()
        day_in_cycle = request.day_in_cycle

        # Preprocess data
        feature_dict = preprocess_data(
            wearable_data=wearable_data,
            hormone_data=hormone_data,
            day_in_cycle=day_in_cycle
        )

        # Run prediction
        result = predict_phase(feature_dict)

        return PredictResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/users")
async def get_users():
    """
    Return list of available user IDs.
    """
    return {"users": [1, 2, 3, 4, 5, 6]}

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.
    """
    try:
        feature_columns = get_feature_columns()
        return ModelInfoResponse(
            model_name="Inner Weather Phase Prediction Model",
            expected_features=feature_columns
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

# Backend API Documentation for Inner Weather

## Overview
The Inner Weather backend is a FastAPI-based REST API that serves machine learning predictions for menstrual phase and mood state. It integrates a pre-trained scikit-learn pipeline to process wearable physiological data and hormone inputs, providing real-time insights for the women's health app.

The backend is designed for stateless, scalable inference, with CORS enabled for frontend integration. It runs on port 8001 and provides Swagger documentation at `/docs`.

## Architecture Overview
The backend follows a modular structure under the `app/` directory:

- **main.py**: Entry point with FastAPI app, CORS middleware, lifespan management, and endpoint definitions.
- **model_loader.py**: Handles loading the joblib artifact and provides access to the pipeline and metadata.
- **schemas.py**: Defines Pydantic models for request/response validation.
- **prediction_service.py**: Core logic for data preprocessing, feature engineering, and inference.
- **mood_mapper.py**: Maps predicted phases to corresponding mood states.
- **__init__.py**: Package initialization.

The app uses asynchronous endpoints for non-blocking I/O, though predictions are synchronous due to scikit-learn's thread-safety.

## Key Technologies
- **Framework**: FastAPI for high-performance async API development.
- **Validation**: Pydantic for automatic request/response serialization and validation.
- **ML Library**: scikit-learn for pipeline execution.
- **Data Processing**: pandas and numpy for feature handling.
- **Serialization**: joblib for model loading.
- **Server**: Uvicorn ASGI server.

## Startup and Lifespan
The app uses FastAPI's lifespan events to load the model once at startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # Loads pipeline, feature_columns, label_classes
    yield
```

This ensures the model is ready before accepting requests and avoids reloading on each prediction.

## Endpoints

### GET /health
**Purpose**: Health check to verify API and model status.

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Implementation**: Checks if the pipeline is loaded; returns 500 if not.

### POST /predict
**Purpose**: Main prediction endpoint for menstrual phase and mood.

**Request Body** (JSON):
```json
{
  "wearable_data": {
    "spo2": 98.4,
    "gsr_mean": 2.1,
    "gsr_phasic_std": 0.07,
    "ppg_rmssd": 42.3,
    "heart_rate": 78,
    "skin_temp": 36.5
  },
  "hormone_data": {
    "estrogen": 120.5,
    "progesterone": 8.3
  },
  "day_in_cycle": 14
}
```

**Response** (JSON):
```json
{
  "predicted_phase": "Luteal",
  "predicted_mood": "Reflective & Deep",
  "confidence": 0.842,
  "probabilities": {
    "Menstrual": 0.05,
    "Follicular": 0.12,
    "Fertility": 0.03,
    "Luteal": 0.842
  }
}
```

**Validation**: Uses Pydantic models to reject malformed requests with 400 errors.

### GET /model/info
**Purpose**: Provides metadata about the loaded model.

**Response**:
```json
{
  "model_name": "Inner Weather Phase Prediction Model",
  "expected_features": ["day_in_study", "cycle_sin_28", ...]
}
```

## Prediction Workflow
The `/predict` endpoint follows this detailed flow:

1. **Request Parsing**: FastAPI automatically validates and parses JSON into `PredictRequest` Pydantic model.

2. **Data Extraction**: Extract wearable_data, hormone_data, and day_in_cycle from request.

3. **Feature Engineering** (in `prediction_service.py`):
   - Calculate cycle_sin_28 = sin(2π * day_in_cycle / 28)
   - Calculate cycle_cos_28 = cos(2π * day_in_cycle / 28)
   - Set stress_score_mean = wearable_data['gsr_mean'] (derived from GSR)
   - Assemble feature_dict:
     ```python
     {
       "day_in_study": day_in_cycle,
       "cycle_sin_28": cycle_sin_28,
       "cycle_cos_28": cycle_cos_28,
       "rmssd_mean": ppg_rmssd,
       "stress_score_mean": stress_score_mean,
       "wrist_temp_mean": skin_temp,
       "lh": None,
       "estrogen": estrogen,
       "pdg": progesterone
     }
     ```

4. **DataFrame Creation**: Convert feature_dict to pandas DataFrame.

5. **Schema Alignment**: Reindex DataFrame to match `feature_columns` from joblib, ensuring column order and handling missing features via imputation.

6. **Inference**:
   - Transform data through pipeline preprocessor.
   - Predict phase: `pipeline.predict(df)[0]`
   - Predict probabilities: `pipeline.predict_proba(df)[0]`
   - Confidence = max(probabilities)

7. **Mood Mapping**: Use `get_mood_from_phase(phase)` to map phase to mood:
   - Menstrual → "Rest & Restore"
   - Follicular → "Light & Energized"
   - Fertility → "Magnetic & Expressive"
   - Luteal → "Reflective & Deep"

8. **Response Construction**: Build `PredictResponse` with phase, mood, confidence, and probabilities dict.

## Error Handling
- **400 Bad Request**: Invalid input data (e.g., missing required fields, wrong types).
- **500 Internal Server Error**: Model loading failures, prediction errors, or unexpected exceptions.
- Exceptions are logged with `traceback.print_exc()` for debugging.

## Security and Best Practices
- **Input Validation**: Pydantic ensures type safety and required fields.
- **CORS**: Enabled for all origins (development-friendly; restrict in production).
- **Stateless Design**: No database; each request is independent.
- **Model Integrity**: Pipeline loaded once; no retraining or modification.
- **Thread Safety**: Model is read-only; suitable for concurrent requests.

## Performance Considerations
- **Startup Time**: Model loading (~1-2 seconds) occurs once.
- **Prediction Latency**: ~50-200ms per request, depending on hardware.
- **Memory Usage**: Joblib file (~100MB) loaded into memory.
- **Scalability**: Can handle multiple concurrent requests; no shared state.

## Configuration
- **Port**: 8001 (configurable in main.py).
- **Model Path**: "phase_prediction_model.joblib" (relative to app directory).
- **CORS Origins**: ["*"] (allow all for development).

## Testing and Debugging
- **Swagger UI**: Access at `http://localhost:8001/docs` for interactive API testing.
- **Health Check**: Use `/health` to verify deployment.
- **Logs**: Startup logs confirm model loading; prediction errors logged to console.

## Integration with Frontend
The backend is designed for seamless integration with the Vite React frontend:
- Frontend sends POST requests to `/predict` with user data.
- Handles JSON responses for UI updates.
- CORS ensures browser compatibility.

## Deployment
- **Local Development**: Run `python -m app.main`.
- **Production**: Use uvicorn or gunicorn with multiple workers.
- **Containerization**: Add Dockerfile for Docker deployment.
- **Environment Variables**: Can extend for configurable model paths.

This backend provides a robust, efficient API for real-time menstrual phase predictions, bridging wearable data with ML insights for the Inner Weather app.

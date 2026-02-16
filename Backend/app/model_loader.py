import joblib
from typing import List, Any

# Global variables to store loaded model components
pipeline = None
feature_columns: List[str] = []
label_classes: List[str] = []

def load_model(model_path: str = "phase_prediction_model.joblib"):
    """
    Load the scikit-learn pipeline and metadata from the joblib file.
    This should be called once at application startup.
    """
    global pipeline, feature_columns, label_classes
    try:
        model_data = joblib.load(model_path)
        pipeline = model_data['pipeline']
        feature_columns = model_data['feature_columns']
        label_classes = model_data['label_classes']
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def get_pipeline():
    """Get the loaded pipeline."""
    if pipeline is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return pipeline

def get_feature_columns():
    """Get the list of expected feature columns."""
    if not feature_columns:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return feature_columns

def get_label_classes():
    """Get the list of label classes."""
    if not label_classes:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return label_classes

import numpy as np
import pandas as pd
from typing import Dict, Any
from .model_loader import get_pipeline, get_feature_columns, get_label_classes
from .mood_mapper import get_mood_from_phase

def preprocess_data(wearable_data: Dict[str, float], hormone_data: Dict[str, float], day_in_cycle: int) -> Dict[str, Any]:
    """
    Preprocess the input data into a feature dictionary aligned with the model's expectations.
    """
    # Calculate sinusoidal features for cycle
    cycle_sin_28 = np.sin(2 * np.pi * day_in_cycle / 28)
    cycle_cos_28 = np.cos(2 * np.pi * day_in_cycle / 28)

    # Derive stress_score_mean from GSR (using gsr_mean as proxy)
    stress_score_mean = wearable_data['gsr_mean']

    # Construct feature dictionary
    feature_dict = {
        "day_in_study": day_in_cycle,
        "cycle_sin_28": cycle_sin_28,
        "cycle_cos_28": cycle_cos_28,
        "rmssd_mean": wearable_data['ppg_rmssd'],
        "stress_score_mean": stress_score_mean,
        "wrist_temp_mean": wearable_data['skin_temp'],
        "lh": None,  # Not provided
        "estrogen": hormone_data['estrogen'],
        "pdg": hormone_data['progesterone']  # progesterone
    }

    return feature_dict

def predict_phase(feature_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference using the loaded pipeline and return prediction results.
    """
    pipeline = get_pipeline()
    feature_columns = get_feature_columns()
    label_classes = get_label_classes()

    # Create DataFrame and align columns
    df = pd.DataFrame([feature_dict])
    df = df.reindex(columns=feature_columns)

    # Run prediction
    prediction = pipeline.predict(df)[0]
    probabilities = pipeline.predict_proba(df)[0]

    # Create probabilities dict
    prob_dict = {label: float(prob) for label, prob in zip(label_classes, probabilities)}
    confidence = float(max(probabilities))

    # Get mood
    mood = get_mood_from_phase(prediction)

    return {
        "predicted_phase": prediction,
        "predicted_mood": mood,
        "confidence": confidence,
        "probabilities": prob_dict
    }

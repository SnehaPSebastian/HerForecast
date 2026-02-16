import os
import pandas as pd
import numpy as np
import joblib
from app.mood_mapper import get_mood_from_phase

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(data_dir, "phase_prediction_model.joblib")
    data_path = os.path.join(data_dir, "synthetic_women_cycle_data.csv")
    output_path = os.path.join(data_dir, "synthetic_predictions.csv")

    # Load model
    artifact = joblib.load(model_path)
    pipeline = artifact['pipeline']
    feature_columns = artifact['feature_columns']
    label_classes = artifact['label_classes']

    # Load synthetic data
    df = pd.read_csv(data_path)
    print(f"Loaded synthetic data: {df.shape}")

    # Rename columns to match expected features
    df = df.rename(columns={
        'Day_in_Cycle': 'day_in_study',
        'Skin_Temperature': 'wrist_temp_mean',
        'PPG_RMSSD': 'rmssd_mean',
        'GSR': 'stress_score_mean',
        'Estrogen': 'estrogen',
        'Progesterone': 'pdg',
        'Phase': 'true_phase'
    })

    # Map true phase labels
    phase_mapping = {
        'menstrual': 'Menstrual',
        'follicular': 'Follicular',
        'ovulation': 'Fertility',
        'luteal': 'Luteal'
    }
    df['true_phase'] = df['true_phase'].map(phase_mapping)

    # Add temporal features
    df['cycle_sin_28'] = np.sin(2 * np.pi * df['day_in_study'] / 28.0)
    df['cycle_cos_28'] = np.cos(2 * np.pi * df['day_in_study'] / 28.0)

    # Prepare features
    X = df.reindex(columns=feature_columns)

    # Predict
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    # Add predictions to dataframe
    df['predicted_phase'] = predictions
    df['confidence'] = np.max(probabilities, axis=1)
    df['predicted_mood'] = [get_mood_from_phase(phase) for phase in predictions]

    # Add probability columns
    for i, label in enumerate(label_classes):
        df[f'prob_{label}'] = probabilities[:, i]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Print summary
    print("\nPrediction Summary:")
    print(df['predicted_phase'].value_counts())
    print(f"\nAverage confidence: {df['confidence'].mean():.3f}")

if __name__ == "__main__":
    main()

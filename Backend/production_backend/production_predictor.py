"""
Production-Ready Web App Predictor V2 with Persistence
Includes SQLite storage, enhanced features, and cycle analytics
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import json
from typing import Dict, List, Optional
from pathlib import Path
from user_history_db import UserHistoryDB

class LHEstimator:
    """Estimate LH when not available"""
    
    @staticmethod
    def estimate_lh(estrogen: float, pdg: float, day_in_cycle: float, temp: float) -> tuple:
        """
        Estimate LH from other hormones
        
        Returns:
            (estimated_lh, confidence_score)
        """
        cycle_position = day_in_cycle % 1.0
        
        # LH peaks around day 14 (0.5 in normalized cycle)
        day_factor = 1.0 - abs(cycle_position - 0.5) * 2
        estrogen_factor = max(0, estrogen)
        pdg_factor = max(0, -pdg)
        temp_factor = max(0, -temp) if cycle_position < 0.5 else max(0, temp)
        
        estimated_lh = (
            0.4 * day_factor +
            0.3 * estrogen_factor +
            0.2 * pdg_factor +
            0.1 * temp_factor
        ) * 2 - 1
        
        # Confidence based on how well features align
        confidence = (day_factor + estrogen_factor) / 2
        
        return estimated_lh, confidence


class ProductionPredictor:
    """
    Production-ready predictor with:
    - SQLite persistence
    - Historical tracking
    - LH estimation
    - Enhanced cycle analytics
    """
    
    CONFIDENCE_THRESHOLDS = {
        'Luteal': 0.60,
        'Fertility': 0.55,
        'Follicular': 0.50,
        'Menstrual': 0.50
    }
    
    def __init__(self, model_dir: str = '.', db_path: str = 'user_history.db'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.db = UserHistoryDB(db_path)
        self.lh_estimator = LHEstimator()
        
        # Load metadata
        with open(self.model_dir / 'model_metadata_final.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.features = self.metadata['features']
        self.classes = self.metadata['classes']
        self.ensemble_weight_lgb = self.metadata['ensemble_weight_lgb']
        
        # Load models
        self._load_models()
        
        print(f"✅ Production Predictor Ready")
        print(f"   Accuracy: {self.metadata['ensemble_accuracy']:.2%}")
        print(f"   Database: {db_path}")
    
    def _load_models(self):
        """Load trained models"""
        from predictor import ImprovedLSTM
        
        self.lgb_model = lgb.Booster(model_file=str(self.model_dir / 'lightgbm_final.txt'))
        
        checkpoint = torch.load(self.model_dir / 'lstm_final.pth', map_location=self.device)
        self.lstm_model = ImprovedLSTM(input_size=checkpoint['input_size']).to(self.device)
        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_model.eval()
    
    def predict(
        self,
        user_id: str,
        date: str,
        features: Dict,
        save_history: bool = True
    ) -> Dict:
        """
        Main prediction method
        
        Args:
            user_id: Unique user identifier
            date: Date in YYYY-MM-DD format
            features: 8 input features from web app
            save_history: Whether to save to database
        
        Returns:
            Prediction with confidence and analytics
        """
        # Handle missing LH
        lh_estimated = False
        lh_confidence = 1.0
        
        if features.get('lh') is None or np.isnan(features.get('lh', np.nan)):
            features['lh'], lh_confidence = self.lh_estimator.estimate_lh(
                features['estrogen'],
                features['pdg'],
                features['day_in_study'],
                features['wrist_temp_mean']
            )
            lh_estimated = True
        
        # Get history
        history = self.db.get_history(user_id, days=21)
        has_history = len(history) >= 3
        
        # Engineer features
        df_engineered = self._engineer_features(features, history)
        X = df_engineered[self.features].values
        
        # Predict
        lgb_proba = self.lgb_model.predict(X)
        
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        with torch.no_grad():
            lstm_output = self.lstm_model(X_tensor)
            lstm_proba = torch.softmax(lstm_output, dim=1).cpu().numpy()
        
        ensemble_proba = (self.ensemble_weight_lgb * lgb_proba + 
                         (1 - self.ensemble_weight_lgb) * lstm_proba)[0]
        
        # Generate prediction
        pred_idx = np.argmax(ensemble_proba)
        predicted_phase = self.classes[pred_idx]
        confidence = float(ensemble_proba[pred_idx])
        threshold = self.CONFIDENCE_THRESHOLDS[predicted_phase]
        
        # Save to database
        if save_history:
            self.db.add_entry(user_id, date, {
                **features,
                'predicted_phase': predicted_phase,
                'confidence': confidence
            })
        
        # Get cycle analytics
        cycle_stats = self.db.get_cycle_stats(user_id)
        
        return {
            'predicted_phase': predicted_phase,
            'confidence': confidence,
            'confidence_threshold': threshold,
            'is_confident': confidence >= threshold,
            'all_probabilities': {
                phase: float(ensemble_proba[i]) 
                for i, phase in enumerate(self.classes)
            },
            'lgb_probabilities': {
                phase: float(lgb_proba[0][i]) 
                for i, phase in enumerate(self.classes)
            },
            'lstm_probabilities': {
                phase: float(lstm_proba[0][i]) 
                for i, phase in enumerate(self.classes)
            },
            'recommendation': self._get_recommendation(predicted_phase, confidence, threshold),
            'analytics': {
                'lh_estimated': lh_estimated,
                'lh_estimation_confidence': lh_confidence if lh_estimated else 1.0,
                'has_history': has_history,
                'history_days': len(history),
                **cycle_stats
            }
        }
    
    def _engineer_features(self, current: Dict, history: List[Dict]) -> pd.DataFrame:
        """Engineer all 116 features"""
        df = pd.DataFrame([current])
        df['id'] = 'user'
        
        # Basic features
        df['cycle_sin_28'] = np.sin(2 * np.pi * (df['day_in_study'] % 28) / 28)
        df['cycle_cos_28'] = np.cos(2 * np.pi * (df['day_in_study'] % 28) / 28)
        df['cycle_sin_14'] = np.sin(2 * np.pi * (df['day_in_study'] % 14) / 14)
        df['cycle_cos_14'] = np.cos(2 * np.pi * (df['day_in_study'] % 14) / 14)
        
        df['estrogen_pdg_ratio'] = df['estrogen'] / (df['pdg'].abs() + 0.1)
        df['pdg_estrogen_ratio'] = df['pdg'] / (df['estrogen'].abs() + 0.1)
        df['lh_estrogen_ratio'] = df['lh'] / (df['estrogen'].abs() + 0.1)
        df['lh_pdg_ratio'] = df['lh'] / (df['pdg'].abs() + 0.1)
        
        df['lh_surge'] = (df['lh'] > 0.5).astype(int)
        df['lh_very_high'] = (df['lh'] > 1.0).astype(int)
        
        df['hormone_sum'] = df['lh'] + df['estrogen'] + df['pdg']
        df['hormone_product'] = df['lh'] * df['estrogen'] * df['pdg']
        
        # Temporal features
        feature_cols = ['wrist_temp_mean', 'rmssd_mean', 'stress_score_mean', 'lh', 'estrogen', 'pdg']
        
        if len(history) >= 3:
            # Use REAL history
            for col in feature_cols:
                hist_values = [h.get(col, 0) for h in history if h.get(col) is not None]
                current_value = df[col].iloc[0]
                
                for window in [3, 7, 14, 21]:
                    if len(hist_values) >= window:
                        df[f'{col}_roll{window}'] = np.mean(hist_values[-window:])
                    else:
                        df[f'{col}_roll{window}'] = current_value
                
                for lag in [1, 3, 7]:
                    if len(hist_values) >= lag:
                        df[f'{col}_lag{lag}'] = hist_values[-lag]
                    else:
                        df[f'{col}_lag{lag}'] = current_value
                
                if len(hist_values) >= 1:
                    df[f'{col}_change1'] = current_value - hist_values[-1]
                else:
                    df[f'{col}_change1'] = 0
                
                if len(hist_values) >= 3:
                    df[f'{col}_change3'] = current_value - hist_values[-3]
                else:
                    df[f'{col}_change3'] = 0
                
                if len(hist_values) >= 7:
                    df[f'{col}_std7'] = np.std(hist_values[-7:])
                else:
                    df[f'{col}_std7'] = 0
        else:
            # Use proxy
            for col in feature_cols:
                for window in [3, 7, 14, 21]:
                    df[f'{col}_roll{window}'] = df[col]
                for lag in [1, 3, 7]:
                    df[f'{col}_lag{lag}'] = df[col]
                df[f'{col}_change1'] = 0
                df[f'{col}_change3'] = 0
                df[f'{col}_std7'] = 0
        
        # Ensure all features exist
        for feature in self.features:
            if feature not in df.columns:
                df[feature] = 0
        
        return df
    
    def _get_recommendation(self, phase: str, confidence: float, threshold: float) -> str:
        """Generate recommendation"""
        if confidence >= threshold + 0.15:
            return f"High confidence - {phase} phase"
        elif confidence >= threshold:
            return f"Moderate confidence - Likely {phase} phase"
        else:
            return f"Low confidence - {phase} phase suggested, manual review recommended"
    
    def get_user_analytics(self, user_id: str) -> Dict:
        """Get comprehensive user analytics"""
        return self.db.get_cycle_stats(user_id)
    
    def export_user_data(self, user_id: str, output_file: str):
        """Export user data"""
        self.db.export_user_data(user_id, output_file)


# Example usage
if __name__ == "__main__":
    predictor = ProductionPredictor('.')
    
    # Simulate user predictions over time
    user_id = "demo_user"
    
    print("\n" + "="*70)
    print("PRODUCTION PREDICTOR WITH PERSISTENCE")
    print("="*70)
    
    # Day 1
    result = predictor.predict(
        user_id=user_id,
        date="2024-01-01",
        features={
            'rmssd_mean': 0.1,
            'wrist_temp_mean': -0.2,
            'estrogen': -0.3,
            'pdg': -0.4,
            'lh': None,  # Will be estimated
            'stress_score_mean': 0.0,
            'oxygen_ratio_mean': 0.0,
            'day_in_study': 0.1
        }
    )
    
    print(f"\nDay 1:")
    print(f"  Phase: {result['predicted_phase']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  LH Estimated: {result['analytics']['lh_estimated']}")
    
    # Simulate more days
    for day in range(2, 15):
        predictor.predict(
            user_id=user_id,
            date=f"2024-01-{day:02d}",
            features={
                'rmssd_mean': 0.1 + day * 0.03,
                'wrist_temp_mean': -0.2 + day * 0.03,
                'estrogen': -0.3 + day * 0.05,
                'pdg': -0.4 + day * 0.04,
                'lh': day * 0.07,
                'stress_score_mean': 0.0,
                'oxygen_ratio_mean': 0.0,
                'day_in_study': 0.1 + day * 0.03
            }
        )
    
    # Day 15 with full history
    result = predictor.predict(
        user_id=user_id,
        date="2024-01-15",
        features={
            'rmssd_mean': 0.5,
            'wrist_temp_mean': 0.2,
            'estrogen': 0.4,
            'pdg': -0.1,
            'lh': 0.9,
            'stress_score_mean': -0.1,
            'oxygen_ratio_mean': 0.0,
            'day_in_study': 0.5
        }
    )
    
    print(f"\nDay 15 (with history):")
    print(f"  Phase: {result['predicted_phase']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  History Days: {result['analytics']['history_days']}")
    
    # Get analytics
    analytics = predictor.get_user_analytics(user_id)
    print(f"\nUser Analytics:")
    for key, value in analytics.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Production predictor working with persistence!")

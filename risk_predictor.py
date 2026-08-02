import os
import joblib
import pandas as pd

class RiskPredictor:
    """
    XGBoost Risk Inference Engine for Phase 2.
    Loads the trained model artifacts from 'Ready dataset/' and infers segment-level risk.
    """
    FEATURE_ORDER = [
        'Pedestrian–vehicle conflicts_ord', 
        'Unsafe merging diverging maneuvers observed_ord', 
        'Junction channelization_ord', 
        'Is sight distance adequate for safe driving?_ord', 
        'Road surface condition_ord', 
        'Roadside commercial activity density_ord', 
        'Frequency of jaywalking pedestrians_ord', 
        'Lane marking condition_ord', 
        'Visibility of pedestrian crossings_ord', 
        'Presence of animals on road_ord'
    ]
    
    # Class mapping
    TIER_DEFINITIONS = {
        "Very Low": {"color": "Green", "emoji": "🟢", "description": "Safe Corridor - Normal Operations"},
        "Low": {"color": "Yellow", "emoji": "🟡", "description": "Acceptable Corridor - Scheduled Maintenance"},
        "High": {"color": "Orange", "emoji": "🟠", "description": "Vulnerable Corridor - Priority Intervention Required"},
        "Very High": {"color": "Red", "emoji": "🔴", "description": "Hazardous Blackspot - Emergency Safety Audit Required"}
    }
    
    def __init__(self, model_dir: str = "../Ready dataset"):
        model_path = os.path.join(model_dir, "best_road_safety_model.pkl")
        classes_path = os.path.join(model_dir, "model_classes.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(classes_path):
            # Fallback check for root relative path
            model_path = os.path.abspath(model_path)
            classes_path = os.path.abspath(classes_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
                
        print(f"Loading trained XGBoost Classifier from: {model_path}")
        self.model = joblib.load(model_path)
        self.classes = joblib.load(classes_path)
        self.feature_order = self.model.get_booster().feature_names

    def predict_risk(self, feature_dict: dict) -> dict:
        """
        Passes a calibrated 10-parameter dictionary to the model.
        Returns predicted risk tier, confidence probabilities, and recommended action.
        """
        # Standardize keys in feature_dict to match booster feature names
        normalized_dict = {}
        booster_features = self.feature_order
        
        for feat in booster_features:
            if feat in feature_dict:
                normalized_dict[feat] = feature_dict[feat]
            else:
                # Fuzzy key match ignoring Unicode dash differences
                matched = False
                for k, v in feature_dict.items():
                    if k.replace('–', '-').replace('—', '-') == feat.replace('–', '-').replace('—', '-'):
                        normalized_dict[feat] = v
                        matched = True
                        break
                if not matched:
                    normalized_dict[feat] = 0
                    
        input_df = pd.DataFrame([normalized_dict])[self.feature_order]
        
        pred_idx = self.model.predict(input_df)[0]
        risk_tier = self.classes[pred_idx]
        
        probs = self.model.predict_proba(input_df)[0]
        prob_dict = {str(self.classes[i]): round(float(probs[i]), 4) for i in range(len(self.classes))}
        
        tier_info = self.TIER_DEFINITIONS.get(risk_tier, {"color": "Gray", "emoji": "⚪", "description": "Unknown"})
        
        return {
            "predicted_risk_tier": risk_tier,
            "emoji": tier_info["emoji"],
            "color": tier_info["color"],
            "description": tier_info["description"],
            "probabilities": prob_dict,
            "input_payload": feature_dict
        }


if __name__ == "__main__":
    from feature_calibrator import FeatureCalibrator
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("Testing RiskPredictor Engine...")
    predictor = RiskPredictor(model_dir="../Ready dataset")
    
    # Test High-Risk Scenario
    test_metrics = {
        "avg_sight_distance_m": 40.0,
        "commercial_count": 6,
        "pedestrian_conflicts": 4,
        "animal_count": 3
    }
    calibrated_payload = FeatureCalibrator.calibrate_segment_metrics(test_metrics)
    result = predictor.predict_risk(calibrated_payload)
    
    print(f"\nPrediction Result: {result['emoji']} [{result['predicted_risk_tier'].upper()}]")
    print(f"Description: {result['description']}")
    print("Class Confidence Probabilities:", result['probabilities'])

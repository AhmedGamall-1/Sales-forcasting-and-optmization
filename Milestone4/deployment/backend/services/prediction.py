import pickle
import pandas as pd
from typing import Dict, Any
from ..models.car import CarFeatures

class PredictionService:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict(self, car_features: CarFeatures) -> Dict[str, Any]:
        try:
            # Convert input features to DataFrame
            features_df = pd.DataFrame([car_features.dict()])
            
            # Make prediction
            prediction = self.model.predict(features_df)
            
            return {
                "predicted_sales": float(prediction[0]),
                "features": car_features.dict()
            }
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}") 
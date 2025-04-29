from fastapi import APIRouter, HTTPException
from ..models.car import CarFeatures, PredictionResponse
from ..services.prediction import PredictionService
from ..config import settings

router = APIRouter()
prediction_service = PredictionService(settings.MODEL_PATH)

@router.post("/predict", response_model=PredictionResponse)
async def predict_sales(car_features: CarFeatures):
    try:
        result = prediction_service.predict(car_features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 
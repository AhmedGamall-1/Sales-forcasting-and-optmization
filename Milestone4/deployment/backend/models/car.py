from pydantic import BaseModel, Field

class CarFeatures(BaseModel):
    year: int = Field(..., description="Year of the car", ge=1900, le=2024)
    mileage: float = Field(..., description="Mileage of the car", ge=0)
    make: str = Field(..., description="Make of the car", min_length=1)
    model: str = Field(..., description="Model of the car", min_length=1)
    price: float = Field(..., description="Price of the car", ge=0)

class PredictionResponse(BaseModel):
    predicted_sales: float
    features: CarFeatures 
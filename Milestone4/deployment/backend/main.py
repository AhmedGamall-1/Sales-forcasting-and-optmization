from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Sales Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize label encoders for categorical variables
label_encoders = {
    'make': LabelEncoder(),
    'model': LabelEncoder(),
    'condition': LabelEncoder(),
    'color': LabelEncoder(),
    'transmission': LabelEncoder(),
    'fuel_type': LabelEncoder()
}

# Create a dummy model with proper feature encoding
def create_dummy_model():
    logger.info("Creating dummy model...")
    # Create dummy data with all features
    dummy_data = pd.DataFrame({
        'year': [2020, 2019, 2018],
        'mileage': [50000, 60000, 70000],
        'make': ['Toyota', 'Honda', 'Ford'],
        'model': ['Camry', 'Civic', 'Fusion'],
        'price': [25000, 22000, 23000],
        'condition': ['good', 'excellent', 'fair'],
        'color': ['black', 'white', 'red'],
        'transmission': ['automatic', 'manual', 'automatic'],
        'fuel_type': ['gasoline', 'hybrid', 'gasoline']
    })
    
    # Fit label encoders
    for col in label_encoders:
        label_encoders[col].fit(dummy_data[col])
    
    # Transform categorical variables
    encoded_data = dummy_data.copy()
    for col in label_encoders:
        encoded_data[col] = label_encoders[col].transform(encoded_data[col])
    
    # Create and fit model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(encoded_data, np.array([25000, 22000, 23000]))
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model

# Load or create model
try:
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
    else:
        model = create_dummy_model()
except Exception as e:
    logger.error(f"Error with model: {str(e)}")
    model = create_dummy_model()

class CarFeatures(BaseModel):
    year: int
    mileage: float
    make: str
    model: str
    price: float
    condition: str = "good"
    color: str = "black"
    transmission: str = "automatic"
    fuel_type: str = "gasoline"

@app.get("/")
async def root():
    return {"message": "Car Sales Prediction API"}

@app.post("/predict")
async def predict_sales(car: CarFeatures):
    try:
        logger.info(f"Received prediction request for {car.make} {car.model}")
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([{
            'year': car.year,
            'mileage': car.mileage,
            'make': car.make,
            'model': car.model,
            'price': car.price,
            'condition': car.condition,
            'color': car.color,
            'transmission': car.transmission,
            'fuel_type': car.fuel_type
        }])
        
        # Transform categorical variables
        encoded_data = input_data.copy()
        for col in label_encoders:
            try:
                encoded_data[col] = label_encoders[col].transform(encoded_data[col])
            except ValueError:
                # If the category is not in the training data, use the most common category
                encoded_data[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
                logger.warning(f"Unknown category for {col}, using default value")
        
        # Make prediction
        prediction = model.predict(encoded_data)[0]
        logger.info(f"Prediction made: {prediction}")
        
        return {
            "prediction": float(prediction),
            "input_features": car.dict()
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8001) 
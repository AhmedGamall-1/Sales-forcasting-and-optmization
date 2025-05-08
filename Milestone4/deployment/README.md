# Car Sales Prediction Application

This is a simple web application that predicts car sales based on various features using a machine learning model.

## Project Structure

```
deployment/
├── backend/
│   └── main.py          # FastAPI backend server
├── frontend/
│   └── app.py           # Streamlit frontend application
└── requirements.txt     # Python dependencies
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python main.py
```
The backend will be available at http://localhost:8000

2. In a new terminal, start the frontend:
```bash
cd frontend
streamlit run app.py
```
The frontend will be available at http://localhost:8501

## Using the Application

1. Open your web browser and go to http://localhost:8501
2. Fill in the car details in the form:
   - Year
   - Mileage
   - Make
   - Model
   - Price
   - Condition
   - Color
   - Transmission
   - Fuel Type
3. Click "Predict Sales" to get a prediction
4. View the results which will show:
   - Predicted sales value
   - Input features (expandable section)

## API Documentation

The backend API documentation is available at http://localhost:8000/docs when the backend server is running.

## Note

Make sure the backend server is running before making predictions from the frontend. 
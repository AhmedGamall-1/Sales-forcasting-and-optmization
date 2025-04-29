import streamlit as st
import requests
import json
import time

# Page config
st.set_page_config(
    page_title="Car Sales Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Car Sales Predictor")
st.markdown("""
This application predicts car sales based on various features. Enter the car details below to get a prediction.
""")

# Create form
with st.form("car_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Year", min_value=1990, max_value=2024, value=2020)
        mileage = st.number_input("Mileage", min_value=0, value=50000)
        make = st.text_input("Make", value="Toyota")
        model = st.text_input("Model", value="Camry")
        price = st.number_input("Price ($)", min_value=0, value=25000)
    
    with col2:
        condition = st.selectbox("Condition", ["excellent", "good", "fair", "poor"])
        color = st.text_input("Color", value="black")
        transmission = st.selectbox("Transmission", ["automatic", "manual"])
        fuel_type = st.selectbox("Fuel Type", ["gasoline", "diesel", "hybrid", "electric"])
    
    submitted = st.form_submit_button("Predict Sales")

# Handle form submission
if submitted:
    try:
        # Prepare data
        car_data = {
            "year": year,
            "mileage": mileage,
            "make": make,
            "model": model,
            "price": price,
            "condition": condition,
            "color": color,
            "transmission": transmission,
            "fuel_type": fuel_type
        }
        
        # Show loading spinner
        with st.spinner('Making prediction...'):
            # Make API request
            response = requests.post("http://localhost:8001/predict", json=car_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success(f"Predicted Sales: ${result['prediction']:,.2f}")
                
                # Show input features in an expandable section
                with st.expander("View Input Features"):
                    st.json(result["input_features"])
                
            else:
                st.error(f"Error: {response.text}")
                st.info("Make sure the FastAPI backend is running on http://localhost:8001")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend server. Please make sure it's running.")
        st.info("Start the backend server by running 'python main.py' in the backend directory")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add footer
st.markdown("""
---
### About
This application uses a machine learning model to predict car sales based on various features.
The model was trained on historical car sales data and uses a Random Forest Regressor algorithm.

### How to Use
1. Fill in the car details in the form above
2. Click 'Predict Sales' to get a prediction
3. View the predicted sales value and input features

### Note
Make sure the backend server is running on http://localhost:8001 before making predictions.
""") 
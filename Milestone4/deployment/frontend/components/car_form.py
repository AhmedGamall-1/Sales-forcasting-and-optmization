import streamlit as st
from typing import Dict, Any, Callable

def create_car_form(on_submit: Callable[[Dict[str, Any]], None]) -> None:
    """Create a form for car feature input."""
    with st.form("car_features_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
            mileage = st.number_input("Mileage", min_value=0, value=50000)
            make = st.text_input("Make", value="Toyota")
        
        with col2:
            model = st.text_input("Model", value="Camry")
            price = st.number_input("Price ($)", min_value=0, value=25000)
        
        submitted = st.form_submit_button("Predict Sales")
        
        if submitted:
            car_data = {
                "year": year,
                "mileage": mileage,
                "make": make,
                "model": model,
                "price": price
            }
            on_submit(car_data) 
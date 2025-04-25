from tkinter import Image
import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

Data = pk.load(open('Carsales_prediction.sav', 'rb'))

st.set_page_config(page_title="Car Sales Prediction", layout="centered")

st.header("🚘 Car Sales Prediction ML Model")
st.info('🔍 Application to predict the price of cars based on various features.')
st.sidebar.header("🏁 Welcome to the Car Sales Prediction App")
st.sidebar.subheader("📊 Predict the price of a car based on its features")
st.sidebar.markdown("This app uses a machine learning model to predict the price of a car based on various features.")

# picture
from PIL import Imagefrom 
from pathlib import Path
import logging


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('car_sales_prediction.log')
stream_handler = logging.StreamHandler()

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

df = pd.read_csv('car_sales_cleaned.csv')
# prediction button
if st.sidebar.button(" Predict Now"):
    try:
        prediction = Data.predict(df)
        logger.info(f"Predicted car price: {prediction[0]:,.2f}")
        st.success(f" **Predicted Car Price:** {prediction[0]:,.2f}")
        st.balloons()  #  ballons effect
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        st.error("Error making prediction")

image_path = Path('car_image.jpg')
if image_path.exists():
    image = Image.open(image_path)
    st.sidebar.markdown("---")
    st.sidebar.image(image, caption="Car Sales Predictor", width=200)
    st.sidebar.markdown("---")

# --- Section 1: Basic Info ---
with st.expander("🧍 Basic Info"):
    gender = st.selectbox('Gender', ['Male', 'Female'])
    annual_income = st.number_input('Annual Income', placeholder='Enter yearly income in USD')
    income_bracket = st.selectbox('Income_Bracket', ['Low', 'Medium', 'High'])
    price_to_income = st.number_input('Price_to_Income', placeholder='e.g., 0.4')
    company = st.text_input('Company', placeholder='e.g., Toyota')
    company_strength = st.selectbox('Company_Strength', ['Low', 'Medium', 'High'])

# --- Section 2: Vehicle Information ---
with st.expander("🚗 Vehicle Information"):
    model = st.text_input('model', placeholder='e.g., Corolla')
    engine = st.text_input('Engine', placeholder='e.g., V6')
    transmission = st.selectbox('Transmission', ['Automatic', 'Manual'])
    color = st.text_input('Color', placeholder='e.g., Red')
    body_style = st.selectbox('Body Style', ['Sedan', 'SUV', 'Coupe', 'Hatchback'])
    engine_to_model = st.number_input('Engine_to_Model', placeholder='e.g., 0.8')
    engine_transmission = st.text_input('Engine_Transmission', placeholder='e.g., V6_Auto')
    pi_plus_model = st.text_input('PI_plus_model', placeholder='e.g., 0.45_Toyota')

# --- Section 3: Dealership Info ---
with st.expander("🏢 Dealership Info"):
    dealer_region = st.text_input('Dealer_Region', placeholder='e.g., East, West')

# --- Section 4: Date and Time ---
with st.expander("📅 Date and Time"):
    year = st.number_input('Year', min_value=1900, max_value=2100, step=1, value=2024)
    month = st.number_input('Month', min_value=1, max_value=12, step=1)
    day = st.number_input('Day', min_value=1, max_value=31, step=1)
    day_of_week = st.selectbox('DayOfWeek', list(range(7)), format_func=lambda x: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][x])
    week_of_year = st.number_input('WeekOfYear', min_value=1, max_value=52)
    year_month = st.text_input('Year_Month', placeholder='e.g., 2024-04')
    is_weekend = st.radio('Is_Weekend', [True, False])

# --- Section 5: Seasonality & Holidays ---
with st.expander("🎉 Seasonality & Holidays"):
    season = st.selectbox('Season', ['Spring', 'Summer', 'Autumn', 'Winter'])
    holiday = st.text_input('Holiday', placeholder='e.g., Eid, Christmas')
    is_holiday = st.radio('Is_Holiday', [True, False])
    seasonal_price_index = st.number_input('Seasonal_Price_Index', placeholder='e.g., 1.1')

# 🧮 Creating the DataFrame for prediction
df = pd.DataFrame({
    'Gender': [gender],
    'Annual_Income': [annual_income],
    'Income_Bracket': [income_bracket],
    'Price_to_Income': [price_to_income],
    'Company': [company],
    'Company_Strength': [company_strength],
    'Model': [model],
    'Engine': [engine],
    'Transmission': [transmission],
    'Color': [color],
    'Body_Style': [body_style],
    'Engine_to_Model': [engine_to_model],
    'Engine_Transmission': [engine_transmission],
    'PI_plus_model': [pi_plus_model],
    'Dealer_Region': [dealer_region],
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'DayOfWeek': [day_of_week],
    'WeekOfYear': [week_of_year],
    'Year_Month': [year_month],
    'Is_Weekend': [is_weekend],
    'Season': [season],
    'Holiday': [holiday],
    'Is_Holiday': [is_holiday],
    'Seasonal_Price_Index': [seasonal_price_index]
})

# prediction button
if st.sidebar.button("🚀 Predict Now"):
    prediction = Data.predict(df)
    st.success(f"💰 **Predicted Car Price:** {prediction[0]:,.2f}")
    st.balloons()  # 🎈 ballons effect

# Retrain_Model.py

import streamlit as st
import pandas as pd
import joblib
import os
from prophet import Prophet
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

from pathlib import Path

# determine project "main" folder regardless of CWD
BASE_DIR = Path(__file__).resolve().parents[1]      # …/main
DATA_PATH = BASE_DIR / "Data" / "Car_sales_Cleand.csv"
MODELS_DIR = BASE_DIR / "Models"

st.set_page_config(page_title="Retrain Forecast Models", layout="wide")
st.title("🔄 Retrain Forecast Models")

# Load historical data
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.groupby("Date")["Price ($)"].mean().reset_index()

# Upload new forecast data
st.subheader("📤 Upload New Forecast Data")
uploaded_file = st.file_uploader("Upload new forecast CSV with 'Date' and 'Price ($)' columns", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.success("New data uploaded successfully!")

    # Show preview
    st.dataframe(new_data)

    # Merge with historical and deduplicate
    full_df = pd.concat([df, new_data]).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    st.subheader("🧾 Combined Dataset")
    st.dataframe(full_df.tail(10))

    if st.button("✅ Retrain All Models"):
        # Save the new dataset
        full_df.to_csv(DATA_PATH, index=False)
        st.success("Updated dataset saved.")

        # --- Retrain Prophet ---
        prophet_df = full_df.rename(columns={"Date": "ds", "Price ($)": "y"})
        prophet_model = Prophet()
        prophet_model.fit(prophet_df)
        joblib.dump(prophet_model, MODELS_DIR / "prophet_model.pkl")
        st.success("✅ Prophet retrained and saved.")

        # --- Retrain XGBoost ---
        xgb_df = full_df.copy()
        xgb_df["year"] = xgb_df["Date"].dt.year
        xgb_df["month"] = xgb_df["Date"].dt.month
        xgb_df["day"] = xgb_df["Date"].dt.day
        xgb_df["dow"] = xgb_df["Date"].dt.dayofweek
        for lag in range(1, 8):
            xgb_df[f"lag_{lag}"] = xgb_df["Price ($)"].shift(lag)
        xgb_df.dropna(inplace=True)
        features = ["year", "month", "day", "dow"] + [f"lag_{i}" for i in range(1, 8)]
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xgb_model.fit(xgb_df[features], xgb_df["Price ($)"])
        joblib.dump(xgb_model, MODELS_DIR / "xgb_forcasting_model.pkl")
        st.success("✅ XGBoost retrained and saved.")

        # --- Retrain ARIMA ---
        arima_series = full_df.set_index("Date")["Price ($)"].asfreq("D").fillna(method="ffill")
        arima_model = ARIMA(arima_series, order=(5,1,0))  # Using default ARIMA(5,1,0)
        arima_model_fit = arima_model.fit()
        joblib.dump(arima_model_fit, MODELS_DIR / "arima_model.pkl")
        st.success("✅ ARIMA retrained and saved.")

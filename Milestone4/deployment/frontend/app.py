import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import os

st.set_page_config(page_title="🚗 Sales Forecasting Dashboard", page_icon="📈", layout="wide")
st.title("🚗 Car Price Forecasting Dashboard")
st.markdown("""
Welcome to the interactive dashboard for car price forecasting! Select a model, view forecasts, and compare results with beautiful visualizations.
---
""")

# --- Sidebar ---
st.sidebar.header("Forecast Controls")
model_choice = st.sidebar.selectbox("Select Forecasting Model", ["Prophet", "LSTM", "XGBoost", "ARIMA", "SARIMA"])
forecast_period = st.sidebar.slider("Days to Forecast", 7, 60, 30)

# --- Load Data ---
data_path = os.path.join(os.path.dirname(__file__), '../../..', 'Milestone1', 'Car_sales_CleanData.csv')
df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values('Date')

st.subheader("Historical Price Data (last 100 days)")
st.line_chart(df[['Date', 'Price ($)']].set_index('Date').tail(100))

# --- Prophet Model ---
def prophet_forecast(df, periods):
    df_prophet = df[['Date', 'Price ($)']].rename(columns={'Date': 'ds', 'Price ($)': 'y'})
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat']].tail(periods), forecast

# --- LSTM Model (Load or Dummy) ---
def lstm_forecast(df, periods):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Price ($)']])
    SEQ_LEN = 30
    def create_sequences(data, seq_length):
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
        return np.array(X)
    last_seq = scaled[-SEQ_LEN:]
    # Dummy: repeat last value (replace with real model for production)
    preds = []
    seq = last_seq.copy()
    for _ in range(periods):
        pred = seq[-1][0]
        preds.append(pred)
        seq = np.vstack([seq[1:], [[pred]]])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=periods)
    return pd.DataFrame({'ds': future_dates, 'yhat': preds}), None

# --- XGBoost Model ---
def xgb_forecast(df, periods):
    df_xgb = df.copy()
    df_xgb['year'] = df_xgb['Date'].dt.year
    df_xgb['month'] = df_xgb['Date'].dt.month
    df_xgb['day'] = df_xgb['Date'].dt.day
    df_xgb['dayofweek'] = df_xgb['Date'].dt.dayofweek
    df_xgb['dayofyear'] = df_xgb['Date'].dt.dayofyear
    df_xgb['weekofyear'] = df_xgb['Date'].dt.isocalendar().week.astype(int)
    for lag in range(1, 8):
        df_xgb[f'lag_{lag}'] = df_xgb['Price ($)'].shift(lag)
    df_xgb = df_xgb.dropna()
    features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear'] + [f'lag_{lag}' for lag in range(1, 8)]
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(df_xgb[features], df_xgb['Price ($)'])
    last_row = df_xgb.iloc[-1]
    preds = []
    future_dates = []
    prev_prices = list(df_xgb['Price ($)'].values[-7:])
    for i in range(periods):
        new_date = last_row['Date'] + pd.Timedelta(days=i+1)
        feat = {
            'year': new_date.year,
            'month': new_date.month,
            'day': new_date.day,
            'dayofweek': new_date.dayofweek,
            'dayofyear': new_date.dayofyear,
            'weekofyear': new_date.isocalendar().week
        }
        for lag in range(1, 8):
            if i < lag:
                feat[f'lag_{lag}'] = prev_prices[-lag + i]
            else:
                feat[f'lag_{lag}'] = preds[-lag]
        X_pred = pd.DataFrame([feat])
        pred = model.predict(X_pred)[0]
        preds.append(pred)
        future_dates.append(new_date)
    return pd.DataFrame({'ds': future_dates, 'yhat': preds}), None

# --- ARIMA Model (Dummy) ---
def arima_forecast(df, periods):
    # Dummy: repeat last value (replace with real ARIMA model for production)
    last_price = df['Price ($)'].iloc[-1]
    future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=periods)
    preds = np.full(periods, last_price)
    return pd.DataFrame({'ds': future_dates, 'yhat': preds}), None

# --- SARIMA Model ---
def sarima_forecast(df, periods):
    import statsmodels.api as sm
    y = df['Price ($)'].copy()
    y.index = pd.DatetimeIndex(df['Date'])
    # Simple SARIMA order (1,1,1)x(1,1,1,12) - can be tuned
    model = sm.tsa.statespace.SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    pred = results.get_forecast(steps=periods)
    forecast = pred.predicted_mean.values
    future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=periods)
    return pd.DataFrame({'ds': future_dates, 'yhat': forecast}), results

# --- Model Selection and Forecast ---
if model_choice == "Prophet":
    forecast_df, _ = prophet_forecast(df, forecast_period)
elif model_choice == "LSTM":
    forecast_df, _ = lstm_forecast(df, forecast_period)
elif model_choice == "XGBoost":
    forecast_df, _ = xgb_forecast(df, forecast_period)
elif model_choice == "ARIMA":
    forecast_df, _ = arima_forecast(df, forecast_period)
elif model_choice == "SARIMA":
    forecast_df, _ = sarima_forecast(df, forecast_period)
else:
    st.error("Invalid model selection.")
    st.stop()

# --- Visualization ---
st.subheader(f"Forecast Results: {model_choice}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'].tail(100), y=df['Price ($)'].tail(100), mode='lines+markers', name='Historical'))
fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines+markers', name=f'{model_choice} Forecast'))
fig.update_layout(title=f"{model_choice} Price Forecast for Next {forecast_period} Days", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Forecast Table")
st.dataframe(forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Price ($)'}).set_index('Date'))

st.markdown("""
---
#### About
This dashboard allows you to forecast car prices using multiple models and visualize the results interactively. For best accuracy, use Prophet or XGBoost. LSTM and ARIMA are illustrative.
""")
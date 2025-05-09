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
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configure main page
st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="wide")
st.title("🚗 Car Price Forecasting 🚗")


@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), './Data/Car_sales_Cleand.csv')
    df = pd.read_csv(data_path, parse_dates=['Date'])
    return df.sort_values('Date')

df = load_data()

# --- Model Loading ---
MODEL_PATH = "./Models/"
def load_model(model_name):
    try:
        return joblib.load(os.path.join(MODEL_PATH, f"{model_name}.pkl"))
    except FileNotFoundError:
        st.error(f"Model file not found: {model_name}.pkl")
        st.stop()

# --- Forecasting Functions ---
def prophet_forecast(periods):
    model = load_model("prophet_model")
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)
def default_forecast(periods):
    """Fallback forecast when errors occur"""
    future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=periods)
    last_value = df['Price ($)'].iloc[-1]
    return pd.DataFrame({'ds': future_dates, 'yhat': [last_value] * periods})

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
    return pd.DataFrame({'ds': future_dates, 'yhat': preds})

def arima_forecast(periods):
        model_fit = joblib.load("./Models/arima_model.pkl")
        forecast = model_fit.forecast(steps=periods)
        future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=periods)
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast,
            'yhat_lower': forecast - 1.96 * model_fit.stderr,
            'yhat_upper': forecast + 1.96 * model_fit.stderr
        })
        future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=periods)
        last_value = df['Price ($)'].iloc[-1]
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': [last_value] * periods
        })

def sarima_forecast(periods):
    try:
        # Use memory-efficient loading
        model_fit = load_model("sarima_model")  # Load the fitted model instead of raw model
        
        # Generate forecast
        forecast = model_fit.forecast(steps=periods)
        future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=periods)
        
        # Create forecast DataFrame
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast,
            'yhat_lower': forecast - 1.96 * model_fit.stderr,
            'yhat_upper': forecast + 1.96 * model_fit.stderr
        })
    except MemoryError:
        st.error("Memory error loading SARIMA model. Try reducing model complexity.")
        # Return a fallback forecast
        future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=periods)
        last_value = df['Price ($)'].iloc[-1]
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': [last_value] * periods
        })


# --- Unified Forecasting ---
def get_forecast(model_choice, periods):
    if model_choice == "Prophet":
        return prophet_forecast(periods)
    elif model_choice == "XGBoost":
        return xgb_forecast(periods)
    elif model_choice == "SARIMA":
        return sarima_forecast(periods)
    elif model_choice == "ARIMA":
        return arima_forecast(periods)
    else:
        st.error("Model not implemented")
        st.stop()

# --- Main Interface ---
st.sidebar.header("Forecast Controls")
model_choice = st.sidebar.selectbox("Select Model", ["Prophet", "XGBoost", "ARIMA", "SARIMA"])
forecast_period = st.sidebar.slider("Days to Forecast", 7, 60, 30)

forecast_df = get_forecast(model_choice, forecast_period)

# --- Visualization ---
st.subheader(f"Forecast Results for model: {model_choice}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines+markers', name=f'{model_choice} Forecast'))
fig.update_layout(title=f"{model_choice} Price Forecast for Next {forecast_period} Days", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

################################
last_hist_date = df['Date'].iloc[-1]
last_hist_value = df['Price ($)'].iloc[-1]
forecast_dates = [last_hist_date] + forecast_df['ds'].tolist()
forecast_values = [last_hist_value] + forecast_df['yhat'].tolist()
st.markdown("---")
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Price ($)'],
    mode='lines', name='Historical',
    line=dict(color='blue')
))

# Forecast, starting from last historical point
fig.add_trace(go.Scatter(
    x=forecast_dates, y=forecast_values,
    mode='lines+markers', name='Forecast',
    line=dict(color='orange', dash='dot')
))

fig.update_layout(
    template="plotly_white",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    showlegend=True
)
#######################################

# Forecast data
if model_choice in ["ARIMA", "SARIMA"]:
    # Add confidence interval for statistical models
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(255,165,0,0.2)'),
        name='Upper CI'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255,165,0,0.2)'),
        name='Lower CI'
    ))

fig.add_trace(go.Scatter(
    x=forecast_df['ds'], y=forecast_df['yhat'],
    mode='lines+markers', name='Forecast',
    line=dict(color='#ff7f0e', width=2, dash='dot')
))

fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    showlegend=True,
    xaxis_title="Date",
    yaxis_title="Price ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- Business Analysis ---
st.subheader("Business Insights")
col1, col2, col3 = st.columns(3)

# Calculate metrics
avg_forecast = forecast_df['yhat'].mean()
last_price = df['Price ($)'].iloc[-1]
change_pct = ((avg_forecast - last_price) / last_price) * 100

with col1:
    st.metric("📈 Average Forecasted Price", 
             f"${avg_forecast:,.2f}",
             help="Average predicted price over forecast period")

with col2:
    st.metric("🔄 Projected Change", 
             f"{change_pct:+.1f}%",
             delta_color="inverse",
             help="Percentage change from last observed price")

with col3:
    st.metric("last price", f"${last_price:,.2f}")


# --- Model Details ---
st.subheader("Model Details")
if model_choice == "ARIMA":
    st.markdown("""
    **ARIMA Model Details**
    - Autoregressive Integrated Moving Average
    - Captures trend and autocorrelation
    - Best for short-term forecasts
    """)
elif model_choice == "SARIMA":
    st.markdown("""
    **SARIMA Model Details**
    - Seasonal ARIMA with seasonal components
    - Handles both trend and seasonality
    - Requires periodic data patterns
    """)
elif model_choice == "Prophet":
    st.markdown("""
    **Prophet Model Details**
    - Additive regression model
    - Automatic seasonality detection
    - Robust to missing data
    """)
else:
    st.markdown("""
    **XGBoost Model Details**
    - Gradient boosting tree model
    - Handles complex non-linear patterns
    - Requires feature engineering
    """)

# --- Forecast Details ---
st.markdown("---")
st.subheader("Detailed Forecast Data")
df_to_show = (
    forecast_df
      .rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
      .set_index('Date')
)
styled = df_to_show.style.format({'Forecast': "${:,.2f}"})
st.dataframe(styled, height=300, use_container_width=True)
# --- Footer ---
st.markdown("---")
st.markdown("""
**About this forecast:**  
Models are retrained weekly using the latest market data.  
Confidence intervals represent 95% probability range for statistical models.
""")
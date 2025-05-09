import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# --- Configuration ---
st.set_page_config(page_title="Car Price Forecasting", page_icon="📈", layout="wide")
st.title("🚗 Car Price Forecasting 🚗")

# --- Caching utilities ---
@st.cache_data
def load_data(path: str = "Milestone4/dep2/main/Data/Car_sales_Cleand.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"] )
    # aggregate duplicates by date
    df = df.groupby("Date")["Price ($)"].mean().reset_index()
    df = df.sort_values("Date").reset_index(drop=True)
    return df

@st.cache_resource
def load_model(filename: str) -> any:
    if not os.path.exists(filename):
        st.error(f"Model file not found: {filename}")
        st.stop()
    try:
        return joblib.load(filename)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Retraining model with statsmodels...")
        # Retrain the model if loading fails
        df = load_data()
        series = df.groupby("Date")["Price ($)"].mean()
        series = series.asfreq("D").fillna(method="ffill")
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit()
        # Save the new model
        joblib.dump(model_fit, filename)
        return model_fit

# --- Forecasting Functions ---
def prophet_forecast(model, periods: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(periods)


def xgb_forecast(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    # feature engineering
    df_feat = df.copy()
    df_feat["year"] = df_feat["Date"].dt.year
    df_feat["month"] = df_feat["Date"].dt.month
    df_feat["day"] = df_feat["Date"].dt.day
    df_feat["dow"] = df_feat["Date"].dt.dayofweek
    # lags
    for lag in range(1, 8):
        df_feat[f"lag_{lag}"] = df_feat["Price ($)"].shift(lag)
    df_feat.dropna(inplace=True)

    features = ["year","month","day","dow"] + [f"lag_{i}" for i in range(1,8)]
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(df_feat[features], df_feat["Price ($)"])

    # rolling predict
    last_date = df_feat["Date"].iloc[-1]
    history = list(df_feat["Price ($)"].iloc[-7:])
    preds, dates = [], []
    for i in range(periods):
        date = last_date + pd.Timedelta(days=i+1)
        feat = {"year":date.year, "month":date.month, "day":date.day, "dow":date.dayofweek}
        for lag in range(1,8):
            feat[f"lag_{lag}"] = history[-lag]
        x = pd.DataFrame([feat])
        p = model.predict(x)[0]
        history.append(p)
        preds.append(p)
        dates.append(date)
    return pd.DataFrame({"ds":dates, "yhat":preds})


def stat_forecast(model, df: pd.DataFrame, periods: int) -> pd.DataFrame:
    # prepare daily series with no duplicates
    series = df.groupby("Date")["Price ($)"].mean()
    series = series.asfreq("D").fillna(method="ffill")

    # attempt forecast with existing model
    try:
        forecast = model.forecast(steps=periods)
        idx = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq="D"
        )
        return pd.DataFrame({
            "ds": idx,
            "yhat": forecast
        })
    except Exception:
        # underfitting fallback: fit new ARIMA model
        model = ARIMA(series, order=(5,1,0))  # Default to ARIMA(5,1,0)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        idx = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq="D"
        )
        return pd.DataFrame({
            "ds": idx,
            "yhat": forecast
        })


# --- Main application ---
def main():
    # load data once
    df = load_data()

    # sidebar controls
    st.sidebar.header("Forecast Controls")
    model_choice = st.sidebar.selectbox("Model", ["Prophet","XGBoost","ARIMA"] )
    periods = st.sidebar.slider("Days to forecast", 7, 60, 30)

    # load models
    prophet_model = load_model("Milestone4/dep2/main/Models/prophet_model.pkl")
    xgb_model = load_model("Milestone4/dep2/main/Models/xgb_forcasting_model.pkl") if model_choice=="XGBoost" else None
    arima_model = load_model("Milestone4/dep2/main/Models/arima_model.pkl") if model_choice in ["ARIMA"] else None

    # compute forecast
    if model_choice=="Prophet":
        forecast_df = prophet_forecast(prophet_model, periods)
    elif model_choice=="XGBoost":
        forecast_df = xgb_forecast(df, periods)
    else:
        forecast_df = stat_forecast(arima_model, df, periods)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines+markers', name=f'{model_choice} Forecast'))
    fig.update_layout(title=f"{model_choice} Price Forecast for Next {periods} Days", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)    
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

  

    # metrics
    avg_f = forecast_df["yhat"].mean()
    last = df["Price ($)"].iloc[-1]
    change = (avg_f-last)/last*100
    c1,c2,c3 = st.columns(3)
    c1.metric("Average Forecast",f"${avg_f:,.2f}")
    c2.metric("Last Price",f"${last:,.2f}")
    c3.metric("% Change",f"{change:.1f}%")

    # data table
    st.subheader("Forecast Data")
    df_show = forecast_df.rename(columns={"ds":"Date","yhat":"Forecast"}).set_index("Date")
    st.dataframe(df_show.style.format({"Forecast":"${:,.2f}"}), height=300)
    # comparison vs last actual price
    last_price = df["Price ($)"].iloc[-1]
    comp = df_show.copy()
    comp['Actual'] = last_price
    comp['Change'] = comp['Forecast'] - comp['Actual']
    comp['Pct Change'] = comp['Change'] / comp['Actual'] * 100
    st.subheader("Forecast vs Actual Comparison")
    st.dataframe(
        comp.style.format({
            'Actual':     "${:,.2f}",
            'Forecast':   "${:,.2f}",
            'Change':     "${:,.2f}",
            'Pct Change': "{:+.1f}%"
        }), height=300
    )

    # model details
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
    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    **About this forecast:**  
    Models are retrained weekly using the latest market data.  
    confidence intervals represent 95% probability range for statistical models.
    """)

if __name__=="__main__":
    main()

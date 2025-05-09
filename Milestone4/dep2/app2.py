import streamlit as st
import requests
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(
    page_title="🚗 New Car Sales Prediction",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 New Car Sales Prediction")
st.markdown("""
Predict future sales of new car models based on market factors, advertising, and pricing strategy.
""")

# Sidebar for additional features
st.sidebar.header("Additional Features")
show_historical = st.sidebar.checkbox("Show Historical Data", value=True)
show_regional = st.sidebar.checkbox("Show Regional Analysis", value=True)
show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)

# Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        make = st.text_input("Car Make", value="Toyota")
        model = st.text_input("Car Model", value="Camry")
        year = st.number_input("Manufacture Year", min_value=2000, max_value=2035, value=2024)
        price = st.number_input("Current Price ($)", min_value=0, value=25000)
        region = st.text_input("Region", value="North")
        ad_spend = st.number_input("Monthly Ad Spend ($)", min_value=0, value=1000)
        discount = st.slider("Discount (%)", 0, 100, 10)

    with col2:
        current_sales = st.number_input("Current Monthly Sales", min_value=0, value=500)
        prediction_type = st.selectbox("Predict by", ["Next X Months", "Specific Future Year"])
        current_year = datetime.datetime.now().year
        if prediction_type == "Next X Months":
            months_ahead = st.slider("How many months ahead?", 1, 60, 6)
            target_year = current_year + (months_ahead // 12)
            target_month = (datetime.datetime.now().month + months_ahead - 1) % 12 + 1
        else:
            target_year = st.number_input("Prediction Year", min_value=current_year, max_value=2100, value=2030)
            target_month = 1  # Assume January

    submitted = st.form_submit_button("🔮 Predict Sales")

# Handle form submission
if submitted:
    input_data = {
        "make": make,
        "model": model,
        "year": year,
        "price": price,
        "region": region,
        "ad_spend": ad_spend,
        "discount": discount,
        "current_sales": current_sales,
        "target_year": target_year,
        "target_month": target_month
    }

    st.info(f"Predicting sales for {make} {model} in {target_month}/{target_year}...")

    try:
        with st.spinner("Generating prediction..."):
            response = requests.post("http://localhost:8001/forecast", json=input_data)

        if response.status_code == 200:
            result = response.json()
            prediction_value = result['forecast']
            st.success(f"📈 Predicted Sales: {prediction_value:,.0f} units in {target_month}/{target_year}")

            # Create prediction/forecast visualization
            if prediction_type == "Next X Months":
                # Generate dates for the prediction period
                dates = pd.date_range(
                    start=datetime.datetime.now(),
                    periods=months_ahead,
                    freq='M'
                )
                # Create prediction values (simple linear projection for demonstration)
                prediction_values = np.linspace(current_sales, prediction_value, months_ahead)
                # Create the prediction chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prediction_values,
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(
                    title=f'Sales Prediction for {make} {model}',
                    xaxis_title='Date',
                    yaxis_title='Predicted Sales',
                    hovermode='x'
                )
                st.plotly_chart(fig,use_container_width=True)

            with st.expander("📋 Input Summary"):
                st.json(result["input_features"])

            # Show feature importance if enabled
            if show_feature_importance:
                st.subheader("⭐ Important Features Affecting Prediction")
                # Mock feature importance values (replace with real values if available)
                feature_importance = pd.DataFrame({
                    'Feature': ['price', 'ad_spend', 'discount', 'current_sales', 'year'],
                    'Importance': [0.35, 0.25, 0.15, 0.15, 0.10]
                }).sort_values('Importance', ascending=True)
                fig_feat = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (Sample)',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_feat, use_container_width=True)

            # Show historical data if enabled
            if show_historical:
                st.subheader("📊 Historical Sales Data")
                # Create sample historical data (replace with actual data from your backend)
                historical_data = pd.DataFrame({
                    'Date': pd.date_range(end=datetime.datetime.now(), periods=12, freq='M'),
                    'Sales': np.random.normal(current_sales, current_sales/4, 12)
                })
                fig_hist = px.line(historical_data, x='Date', y='Sales',
                                 title='Historical Sales Trend')
                st.plotly_chart(fig_hist, use_container_width=True)

            # Show regional analysis if enabled
            if show_regional:
                st.subheader("🌍 Regional Sales Analysis")
                # Create sample regional data (replace with actual data from your backend)
                regions = ['North', 'South', 'East', 'West']
                regional_data = pd.DataFrame({
                    'Region': regions,
                    'Sales': np.random.normal(prediction_value, prediction_value/4, len(regions))
                })
                fig_regional = px.bar(regional_data, x='Region', y='Sales',
                                    title='Sales by Region')
                st.plotly_chart(fig_regional, use_container_width=True)

        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the backend server. Make sure it's running on http://localhost:8001")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

# Footer
st.markdown("""
---
### About
This app predicts sales for new cars based on:
- Car details (make, model, year, price)
- Market region
- Monthly ad spend and discounts
- Current sales numbers

**Ensure the backend API is available at `http://localhost:8001/forecast`.**
""")
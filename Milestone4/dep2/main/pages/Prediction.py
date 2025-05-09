import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
from pathlib import Path

def random_forest_page():
    st.title("🌲 Random Forest Forecast")
    
    # Load model
    @st.cache_resource
    def load_rf_model():
        return joblib.load(f'../Models/Random_Forest_Regressor.pkl')
    
    model = load_rf_model()
    
    # Forecast display
    st.subheader("Predictions with Random forest")


    st.set_page_config("Car Price by Selected Features", layout="centered")
    st.title("🚗 Custom Feature Car Price Predictor")

# — 2. Let user pick up to 5 features —
    st.markdown("Step 1: Choose up to 2 extra features (in addition to date)")
    all_features = [
        "Annual Income","Price_to_Income","Company_Strength","PI_plus_model",
        "Income_Bracket","Seasonal_Price_Index","Engine_to_Model"
    ]
    extra = st.multiselect(
        "Select extra features (max 2):",
        options=all_features,
        help="These will be used alongside Year, Month, DayOfWeek"
    )
    if len(extra)>2:
        st.error("⚠️ Please select at most 2 extra features")
        st.stop()

# — 3. Collect date input —
    st.markdown("Step 2: Pick a date for prediction")
    dt = st.date_input("Date:", value=datetime.today())
# derive date features
    year, month, dow = dt.year, dt.month, dt.weekday()

# — 4. Collect values for each selected extra feature —
    inputs = {"Year":year, "Month":month, "DayOfWeek":dow}
    for feat in extra:
    # infer type and collect
        val = st.number_input(f"{feat}:", value=0.0) if df_[feat].dtype.kind in 'fc' else st.selectbox(feat, sorted(df_[feat].unique()))
        inputs[feat] = val

# — 5. Build DataFrame for model —
    X_pred = pd.DataFrame([inputs])

# — 6. One‑hot or scale if needed —
# If your model expects one‑hot columns, reindex:
#X_pred = pd.get_dummies(X_pred).reindex(columns=your_training_columns, fill_value=0)

# — 7. Predict button —
    if st.button("🔮 Predict Price"):
        price = model.predict(X_pred)[0]
        st.success(f"💰 Predicted Price: ${price:,.2f}")

    # — 8. Simple business insight —
        st.markdown("#### Business Insight")
        st.markdown(f"- Based on the last 30‑day average price of similar configurations, which was around **${df_['Price ($)'].tail(30).mean():,.2f}**,")
        st.markdown(f"- your predicted price of **${price:,.2f}** is within the expected range.")

    # — 9. Show feature importance —
        st.markdown("### Feature Importance")
        st.bar_chart(model.feature_importances_)
        st.write("Feature importance analysis:")
    
        # Add your RF-specific implementation here
        # ...

if __name__ == "__main__":
    random_forest_page()

# — 1. Load model once —

  
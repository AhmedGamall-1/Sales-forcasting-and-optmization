# pages/02_Car_Price_Prediction.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Car Price Prediction", page_icon="🚘", layout="centered")
st.title("🚘 Predict Car Price with Random Forest")

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_PATH  = BASE_DIR / "Data" / "dataModeling.csv"
MODEL_PATH = BASE_DIR / "Models" / "Random_Forest_Regressor.pkl"

# ─── Load Data & Model ─────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Print column names for debugging
    st.write("Available columns:", df.columns.tolist())
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()
    m = joblib.load(MODEL_PATH)
    # fetch feature names if available
    feat_names = getattr(m, "feature_names_in_", None)
    n_feat     = getattr(m, "n_features_in_", None)
    return m, feat_names, n_feat

df = load_data()
model, feature_names_in, n_features_in = load_model()

# ─── Input Controls (main canvas) ─────────────────────────────────────────────
st.subheader("Enter Car Features for Prediction")

annual_income = st.number_input(
    "Annual Income ($)",
    min_value=int(df["Annual Income"].min()),
    max_value=int(df["Annual Income"].max()),
    value=int(df["Annual Income"].median())
)

engine = st.selectbox(
    "Engine (type)",
    df["Engine"].astype(str).unique().tolist()
)

# Get years from Date column
years = sorted(df["Date"].dt.year.unique())
year = st.selectbox("Year", years)
transmission = st.selectbox("Transmission", df["Transmission"].unique())
body_style   = st.selectbox("Body Style", df["Body Style"].unique())
company      = st.selectbox("Company", df["Company"].unique())
dealer_reg   = st.selectbox("carsales Region", df["Dealer_Region"].unique())
month       = st.selectbox("Month", df["Month"].unique())
is_weekend = st.selectbox("Is Weekend", ["No","Yes"])
is_weekend_flag = 1 if is_weekend=="Yes" else 0

price_to_income = st.number_input(
    "Price_to_Income Ratio",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.01
)

# ─── Build raw DataFrame ────────────────────────────────────────────────────────
input_dict = {
    "Annual Income": annual_income,
    "Engine":         engine,
    "Year":           year,
    "Transmission":   transmission,
    "Body Style":     body_style,
    "Company":        company,
    "carsales Region":  dealer_reg,
    "Month":          month,
    "Is_Weekend":     is_weekend,
    "Price_to_Income":price_to_income
}
X_raw = pd.DataFrame([input_dict])
st.subheader("Input Features")
st.table(X_raw.T.rename(columns={0:"Value"}))

# ─── Preprocess to match training ──────────────────────────────────────────────
# one-hot encode raw
X_dummies = pd.get_dummies(X_raw)

# If model recorded feature names, use them to subset/reorder
if feature_names_in is not None:
    missing = set(feature_names_in) - set(X_dummies.columns)
    if missing:
        st.error(f"Cannot predict: missing features {missing}")
        st.stop()
    X_proc = X_dummies[feature_names_in]
# Otherwise fall back to numeric slice
else:
    X_proc = X_dummies.iloc[:, :n_features_in]

# ─── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Price"):
    try:
        pred = model.predict(X_proc)[0]
        st.success(f"🛒 **Estimated Price:** ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

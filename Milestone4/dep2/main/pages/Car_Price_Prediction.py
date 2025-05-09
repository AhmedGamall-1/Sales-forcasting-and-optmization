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

def create_income_bracket(income):
    if income < 50000:
        return "Low"
    elif income < 100000:
        return "Medium"
    else:
        return "High"

def get_seasonal_index(month):
    # Simple seasonal index based on month
    if month in [12, 1, 2]:  # Winter
        return 0.9
    elif month in [3, 4, 5]:  # Spring
        return 1.0
    elif month in [6, 7, 8]:  # Summer
        return 1.1
    else:  # Fall
        return 0.95

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

# Create income bracket
income_bracket = create_income_bracket(annual_income)

# Engine selection
engine_types = ["V6", "V8", "I4", "I6", "Electric"]
engine = st.selectbox("Engine (type)", engine_types)

# Get years from Date column
years = sorted(df["Date"].dt.year.unique())
year = st.selectbox("Year", years)

# Model selection
models = ["Sedan", "SUV", "Truck", "Coupe", "Hatchback"]
model = st.selectbox("Model", models)

# Use default options for missing columns
transmission = st.selectbox("Transmission", ["Automatic", "Manual"])

body_style = st.selectbox("Body Style", ["Sedan", "SUV", "Truck", "Coupe", "Hatchback"])

company = st.selectbox("Company", ["Toyota", "Honda", "Ford", "BMW", "Mercedes"])

# Use default regions
dealer_reg = st.selectbox("carsales Region", ["North", "South", "East", "West"])

# Get month from Date column
month = st.selectbox("Month", range(1, 13))

# Calculate seasonal price index
seasonal_index = get_seasonal_index(month)

is_weekend = st.selectbox("Is Weekend", ["No","Yes"])
is_weekend_flag = 1 if is_weekend=="Yes" else 0

is_holiday = st.selectbox("Is Holiday", ["No","Yes"])
is_holiday_flag = 1 if is_holiday=="Yes" else 0

price_to_income = st.number_input(
    "Price_to_Income Ratio",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.01
)

# Company strength (simplified)
company_strength = {
    "Toyota": 0.8,
    "Honda": 0.7,
    "Ford": 0.6,
    "BMW": 0.9,
    "Mercedes": 0.9
}[company]

# ─── Build raw DataFrame ────────────────────────────────────────────────────────
input_dict = {
    "Annual Income": annual_income,
    "Engine": engine,
    "Year": year,
    "Transmission": transmission,
    "Body Style": body_style,
    "Company": company,
    "carsales Region": dealer_reg,
    "Month": month,
    "Is_Weekend": is_weekend_flag,
    "Is_Holiday": is_holiday_flag,
    "Price_to_Income": price_to_income,
    "Income_Bracket": income_bracket,
    "Seasonal_Price_Index": seasonal_index,
    "Company_Strength": company_strength,
    "model": model,
    "Engine_to_Model": f"{engine}_{model}",
    "PI_plus_model": f"{price_to_income}_{model}",
    "Holiday": is_holiday
}

X_raw = pd.DataFrame([input_dict])

# ─── Preprocess to match training ──────────────────────────────────────────────
# Create all possible categorical combinations
categorical_cols = ['Engine', 'model', 'Body Style', 'Company', 'Transmission', 
                   'carsales Region', 'Income_Bracket', 'Engine_to_Model', 
                   'PI_plus_model', 'Holiday']

# One-hot encode categorical variables
X_dummies = pd.get_dummies(X_raw, columns=categorical_cols)

# If model recorded feature names, use them to subset/reorder
if feature_names_in is not None:
    # Print available features for debugging
    st.write("Available features after encoding:", X_dummies.columns.tolist())
    st.write("Model expects features:", feature_names_in)
    
    missing = set(feature_names_in) - set(X_dummies.columns)
    if missing:
        st.error(f"Cannot predict: missing features {missing}")
        st.stop()
    X_proc = X_dummies[feature_names_in]
else:
    X_proc = X_dummies.iloc[:, :n_features_in]

# ─── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Price"):
    try:
        pred = model.predict(X_proc)[0]
        st.success(f"🛒 **Estimated Price:** ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Debug info:")
        st.write("Input shape:", X_proc.shape)
        st.write("Model feature count:", n_features_in)

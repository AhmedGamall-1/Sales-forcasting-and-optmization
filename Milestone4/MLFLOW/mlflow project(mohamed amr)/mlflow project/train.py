# ------------------------------ Import Libraries ------------------------------ #
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# ------------------------------ ARIMA Wrapper ------------------------------ #
class ArimaWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, arima_model_fit):
        self.arima_model_fit = arima_model_fit

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        steps = int(model_input.iloc[0]["steps"])
        forecast = self.arima_model_fit.forecast(steps=steps)
        return pd.DataFrame({"forecast": forecast})


# ------------------------------ Load Data ------------------------------ #
def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=['Date', 'Model'], inplace=True, errors='ignore')
    df = df[df['Price ($)'].notna()]
    return df


# ------------------------------ Preprocessor ------------------------------ #
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    if 'Price ($)' in numeric_cols:
        numeric_cols.remove('Price ($)')

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])


# ------------------------------ Get Model ------------------------------ #
def get_model(model_name, n_estimators, max_depth):
    if model_name == 'XGBoost':
        return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_name == 'RandomForest':
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_name == 'LinearRegression':
        return LinearRegression()
    elif model_name == 'DecisionTree':
        return DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    elif model_name == 'KNN':
        return KNeighborsRegressor(n_neighbors=5)
    elif model_name == 'ARIMA':
        return 'ARIMA'
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


# ------------------------------ Train ARIMA ------------------------------ #
def train_arima(data_path, arima_order):
    df = load_data(data_path)
    ts = df['Price ($)']

    # تقسيم السلسلة الزمنية إلى train/test (80% train, 20% test)
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    # تدريب الموديل على بيانات التدريب فقط
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()

    # التنبؤ على بيانات الاختبار
    forecast = model_fit.forecast(steps=len(test))

    # حساب المقاييس
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    try:
        r2 = r2_score(test, forecast)
    except Exception:
        r2 = float('nan')
    aic = model_fit.aic
    bic = model_fit.bic

    # حفظ الموديل
    with open("arima_model.pkl", "wb") as f:
        pickle.dump(model_fit, f)

    mlflow.set_experiment("Car_Price_ARIMA_New")

    with mlflow.start_run():
        mlflow.log_param("arima_order", arima_order)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2_Score", r2)
        mlflow.log_metric("AIC", aic)
        mlflow.log_metric("BIC", bic)
        mlflow.log_artifact("arima_model.pkl")

        input_example = pd.DataFrame({"steps": [5]})
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ArimaWrapper(model_fit),
            artifacts={"arima_model": os.path.abspath("arima_model.pkl")},
            input_example=input_example
        )

        print("\n-")
        print("✅ ARIMA model logged successfully to MLflow")
        print(f"📊 RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}, AIC = {aic:.2f}, BIC = {bic:.2f}")
        print("-\n")


# ------------------------------ Train Regression Model ------------------------------ #
def train_model(data_path, model_name, n_estimators, max_depth):
    if model_name == 'ARIMA':
        raise ValueError("Use train_arima() instead for ARIMA models.")

    df = load_data(data_path)
    X = df.drop(columns=['Price ($)'])
    y = df['Price ($)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(X)
    model = get_model(model_name, n_estimators, max_depth)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    mlflow.set_experiment("Car_Price_Regression")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model", model_name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2_Score", r2)

        mlflow.sklearn.log_model(pipeline, "model")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"{model_name} - Actual vs Predicted")
        plt.grid(True)
        plt.savefig("actual_vs_pred.png")
        mlflow.log_artifact("actual_vs_pred.png")
        plt.close()

        print("=" * 60)
        print(f"✅ Model: {model_name} logged to MLflow successfully")
        print(f"📊 RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}")
        print("=" * 60)


# ------------------------------ Run All Trials ------------------------------ #
def run_all_trials(data_path):
    trials = [
        ("RandomForest", 50, 5),
        ("RandomForest", 100, 10),
        ("RandomForest", 150, 15),
        ("RandomForest", 200, 20),
        ("RandomForest", 250, 25),
        ("RandomForest", 300, 30),
        ("RandomForest", 350, 35),
        ("XGBoost", 50, 5),
        ("XGBoost", 100, 10),
        ("XGBoost", 150, 15),
        ("XGBoost", 200, 20),
        ("XGBoost", 250, 25),
        ("XGBoost", 300, 30),
        ("XGBoost", 350, 35),
        ("LinearRegression", 0, 0),
        ("DecisionTree", 0, 5),
        ("DecisionTree", 0, 10),
        ("DecisionTree", 0, 15),
        ("KNN", 0, 0)
    ]

    arima_orders = [
        (1, 1, 0),
        (2, 1, 1),
        (3, 1, 0),
        (5, 1, 0),
        (7, 1, 2),
        (8, 2, 1),
        (10, 1, 0)
    ]

    for model, n_est, max_d in trials:
        print(f"\n🚀 Running trial: {model} | n_estimators={n_est}, max_depth={max_d}")
        train_model(data_path, model, n_est, max_d)

    for order in arima_orders:
        print(f"\n📈 Running ARIMA trial with order={order}")
        train_arima(data_path, order)


# ------------------------------ Predict Function ------------------------------ #
def predict(data_path, model_path, model_type, steps=None):
    df = load_data(data_path)

    if model_type.lower() == 'arima':
        if steps is None:
            raise ValueError("Please provide 'steps' parameter for ARIMA prediction.")
        model = mlflow.pyfunc.load_model(model_path)
        input_df = pd.DataFrame({"steps": [steps]})
        forecast_df = model.predict(input_df)
        print(f"\n📈 ARIMA forecast for next {steps} steps:")
        print(forecast_df)
        return forecast_df

    else:
        X = df.drop(columns=['Price ($)'])
        model = mlflow.sklearn.load_model(model_path)
        y_pred = model.predict(X)
        df['Predicted Price'] = y_pred
        print(f"\n🔮 Predictions using {model_type} model:")
        print(df[['Predicted Price']])
        return df[['Predicted Price']]


# ------------------------------ Main ------------------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and Predict Car Price Models including ARIMA.")
    parser.add_argument('--mode', type=str, choices=['train_all', 'predict'], default='train_all',
                        help="Mode: 'train_all' to train all models, 'predict' to make predictions.")
    parser.add_argument('--data_path', type=str, default='Car_sales_Cleand.csv', help="Path to dataset CSV file.")
    parser.add_argument('--model_path', type=str, default=None, help="Path to saved model for prediction.")
    parser.add_argument('--model_type', type=str, choices=['XGBoost', 'RandomForest', 'LinearRegression', 'DecisionTree', 'KNN', 'ARIMA'],
                        default='XGBoost', help="Model type for prediction.")
    parser.add_argument('--steps', type=int, default=None, help="Number of steps to forecast for ARIMA.")

    args = parser.parse_args()

    if args.mode == 'train_all':
        run_all_trials(args.data_path)
    elif args.mode == 'predict':
        if args.model_path is None:
            raise ValueError("Please provide --model_path for prediction mode.")
        predict(args.data_path, args.model_path, args.model_type, args.steps)

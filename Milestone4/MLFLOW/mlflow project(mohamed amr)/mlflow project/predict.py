# ------------------------------ Import Libraries ------------------------------ #
import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
from xgboost import XGBRegressor 

warnings.filterwarnings("ignore")


# ------------------------------ Load and Clean Data ------------------------------ #
def load_data(path):
    "Loading the dataset and cleaning it by dropping unnecessary columns."
    df = pd.read_csv(path)  # Loading the data from csv
    df.drop(columns=['Date', 'Model'], inplace=True, errors='ignore')  # Drop unneeded columns
    df = df[df['Price ($)'].notna()]  # Drop rows where target 'Price' is missing
    return df


# ------------------------------ Preprocessing Pipeline ------------------------------ #
def build_preprocessor(X):
    "Build preprocessing pipeline for numeric and categorical columns."
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    if 'Price ($)' in numeric_cols:
        numeric_cols.remove('Price ($)')  # Don't preprocess the target column

    # Numeric preprocessing: impute missing values and scale the features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
        ('scaler', StandardScaler())  # Standardize features
    ])

    # Categorical preprocessing: impute missing values and one-hot encode the features
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent category
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))  # One-hot encode
    ])

    return ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])


# ------------------------------ Load Model and Predict ------------------------------ #
def predict(data_path, model_path):
    "Loading the trained model and useing it to make predictions."
    # Load the test data
    df = load_data(data_path)
    X = df.drop(columns=['Price ($)'])

    # Load the model from MLflow
    #model = mlflow.sklearn.load_model(model_path)
    model = mlflow.pyfunc.load_model(model_path)
    # Make predictions
    y_pred = model.predict(X)

    # Return the predictions
    df['Predicted Price'] = y_pred
    return df[['Predicted Price']]


# ------------------------------ Command Line Interface (CLI) ------------------------------ #
if __name__ == "__main__":

    # Define the default paths
    DEFAULT_MODEL_PATH = r"mlruns/568828146479296419/70f2e95976c74f67a1d1539c64ef7d58/artifacts/saved_model"
    DEFAULT_CSV_PATH = 'Car_sales_Cleand.csv'  # Default CSV path

    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default=DEFAULT_CSV_PATH,  # Default to the specified CSV path
        help="Path to the dataset for making predictions"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_PATH,  # Default to the saved model path
        help="Path to the trained model"
    )
    args = parser.parse_args()

    # Make predictions with the provided arguments or default values
    predictions = predict(args.data_path, args.model_path)
    print(predictions)
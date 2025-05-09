import argparse
import pandas as pd
import mlflow.pyfunc
import matplotlib.pyplot as plt

def predict_arima(model_path, steps):
    """
    function to load an ARIMA model from MLflow and predict the next steps.
    """
    # Load the model from MLflow
    model = mlflow.pyfunc.load_model(model_path)
    
    # prepare input data
    input_df = pd.DataFrame({"steps": [steps]})
    
    # predict the next steps
    forecast_df = model.predict(input_df)
    
    print(f"\nARIMA forecast for next {steps} steps:")
    print(forecast_df)
    
    # plotting the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df['forecast'], marker='o', linestyle='-', color='blue')
    plt.title(f'ARIMA Forecast for Next {steps} Steps')
    plt.xlabel('Step')
    plt.ylabel('Forecasted Value')
    plt.grid(True)
    plt.show()
    
    return forecast_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with ARIMA model logged in MLflow and plot forecast.")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path or URI to the MLflow ARIMA model (e.g. 'runs:/<run_id>/model')"
    )
    parser.add_argument(
        '--steps',
        type=int,
        required=True,
        help="Number of future steps to forecast"
    )
    args = parser.parse_args()

    predict_arima(args.model_path, args.steps)

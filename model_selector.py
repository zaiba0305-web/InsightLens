
import pandas as pd

from models.arima_model import train_arima
from models.prophet_model import train_prophet
from models.xgboost_model import train_xgboost
from models.lstm_model import train_lstm

def select_best_model(data_path):
    
    results = {}

    print("Training ARIMA...")
    arima_forecast = train_arima(data_path)
    results["arima"] = arima_forecast["forecast"].mean()

    print("Training Prophet...")
    prophet_forecast = train_prophet(data_path)
    results["prophet"] = prophet_forecast["forecast"].mean()

    print("Training XGBoost...")
    xgb_forecast = train_xgboost(data_path)
    results["xgboost"] = xgb_forecast["forecast"].mean()

    print("Training LSTM...")
    lstm_forecast = train_lstm(data_path)
    results["lstm"] = lstm_forecast["forecast"].mean()

    best_model = min(results, key=results.get)

    print("\nModel comparison:")
    for model, score in results.items():
        print(f"{model}: {score:.2f}")

    print(f"\nBest model: {best_model}")

    return best_model, results

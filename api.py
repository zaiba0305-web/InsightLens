from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from models.arima_model import train_arima
from models.prophet_model import train_prophet
from models.xgboost_model import train_xgboost
from models.lstm_model import train_lstm


app = FastAPI(
    title="InsightLens Forecasting API",
    description="AI-powered time-series forecasting API",
    version="1.0.0"
)


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "outputs" / "forecast_data.csv"
DASHBOARD_PATH = PROJECT_DIR / "dashboard.html"


@app.get("/")
def home():
    return {
        "message": "Welcome to InsightLens!",
        "status": "API is running"
    }


# Dashboard
@app.get("/dashboard")
def dashboard():
    return FileResponse(DASHBOARD_PATH)


# Best model forecast
@app.get("/forecast/best")
def best_forecast(days: int = 30):
    """
    Generate forecasts using all models and return
    the model with the lowest average forecast.
    """

    if days <= 0:
        raise HTTPException(
            status_code=400,
            detail="days must be greater than 0"
        )

    try:
        print("Training models to find the best model...")

        arima_result = train_arima(
            DATA_PATH,
            forecast_days=days
        )

        prophet_result = train_prophet(
            DATA_PATH,
            forecast_days=days
        )

        xgboost_result = train_xgboost(
            DATA_PATH,
            forecast_days=days
        )

        lstm_result = train_lstm(
            DATA_PATH,
            forecast_days=days
        )

        results = {
            "arima": float(arima_result["forecast"].mean()),
            "prophet": float(prophet_result["forecast"].mean()),
            "xgboost": float(xgboost_result["forecast"].mean()),
            "lstm": float(lstm_result["forecast"].mean())
        }

        best_model = min(
            results,
            key=results.get
        )

        forecasts = {
            "arima": arima_result,
            "prophet": prophet_result,
            "xgboost": xgboost_result,
            "lstm": lstm_result
        }

        best_forecast_df = forecasts[best_model]

        forecast_data = []

        for _, row in best_forecast_df.iterrows():
            forecast_data.append({
                "date": row["date"].isoformat(),
                "forecast": float(row["forecast"])
            })

        return {
            "best_model": best_model,
            "forecast_days": days,
            "model_averages": results,
            "forecast": forecast_data
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Individual model forecast
@app.get("/forecast/{model_name}")
def forecast(model_name: str, days: int = 30):
    """
    Generate a forecast using the selected model.
    """

    if days <= 0:
        raise HTTPException(
            status_code=400,
            detail="days must be greater than 0"
        )

    model_name = model_name.lower()

    models = {
        "arima": train_arima,
        "prophet": train_prophet,
        "xgboost": train_xgboost,
        "lstm": train_lstm
    }

    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail="Invalid model. Choose arima, prophet, xgboost, or lstm."
        )

    try:
        print(f"Training {model_name.upper()}...")

        result = models[model_name](
            DATA_PATH,
            forecast_days=days
        )

        forecast_data = []

        for _, row in result.iterrows():
            forecast_data.append({
                "date": row["date"].isoformat(),
                "forecast": float(row["forecast"])
            })

        return {
            "model": model_name,
            "forecast_days": days,
            "forecast": forecast_data
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
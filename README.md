# InsightLens – AI-Powered Time-Series Forecasting

InsightLens is an AI-powered time-series forecasting system that compares
multiple machine learning and statistical forecasting models and
automatically selects the best-performing model.

## Features

- Time-series data generation
- Feature engineering
- ARIMA forecasting
- Prophet forecasting
- XGBoost forecasting
- LSTM forecasting
- Automatic model comparison
- Automatic best-model selection
- FastAPI REST API
- Interactive forecasting dashboard
- Forecast visualization using charts
- Configurable forecast horizon

## Models Used

| Model | Type |
|---|---|
| ARIMA | Statistical Time-Series |
| Prophet | Time-Series Forecasting |
| XGBoost | Machine Learning |
| LSTM | Deep Learning |

## How It Works

1. Time-series data is prepared.
2. Features are generated.
3. Four forecasting models generate predictions.
4. The average forecast from each model is calculated.
5. The models are compared.
6. The model with the lowest average forecast is selected.
7. The selected model's forecast is displayed on the dashboard.

## API Endpoints

### Home

GET `/`

Checks whether the API is running.

### Individual Forecast

GET `/forecast/{model_name}?days=30`

Supported models:

- arima
- prophet
- xgboost
- lstm

### Best Forecast

GET `/forecast/best?days=30`

Runs all forecasting models and returns the selected best model
along with model comparison values and forecast results.

## Dashboard

The dashboard provides:

- Forecast day selection
- Best model display
- Model comparison
- Forecast visualization
- Forecast results table

## Technologies

- Python
- FastAPI
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Prophet
- TensorFlow / Keras
- HTML
- CSS
- JavaScript
- Chart.js

## Running the Project

Activate the virtual environment and start the FastAPI server:

```bash
uvicorn api:app --reload
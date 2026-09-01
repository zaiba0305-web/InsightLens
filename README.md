# InsightLens – AI-Powered Time-Series Forecasting

InsightLens is an AI-powered time-series forecasting system that predicts future values using multiple machine learning and statistical forecasting models.

The system trains four different forecasting models — ARIMA, Prophet, XGBoost, and LSTM — compares their forecast results, automatically selects the model with the lowest average forecast, and displays the results through an interactive web dashboard.

## 🚀 Features

- Time-series data forecasting
- Multiple forecasting models
- Automatic model comparison
- Best model selection
- 30-day forecast generation
- REST API using FastAPI
- Interactive forecasting dashboard
- Forecast visualization using charts
- JSON-based API responses

## 🤖 Forecasting Models

### ARIMA
A statistical time-series forecasting model used to capture trends and patterns in sequential data.

### Prophet
A forecasting model designed to handle time-series trends and seasonality.

### XGBoost
A gradient boosting machine-learning model used for forecasting using engineered time-series features.

### LSTM
A deep-learning recurrent neural network designed to learn patterns from sequential data.

## 🏆 Model Selection

InsightLens trains all four models and calculates the average forecast produced by each model.

The model with the lowest average forecast is automatically selected as the best model.

Example:

| Model | Average Forecast |
|-------|-----------------:|
| ARIMA | 186.36 |
| Prophet | 184.63 |
| XGBoost | 105.79 |
| LSTM | 188.16 |

**Best Model: XGBoost**

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- FastAPI
- Uvicorn
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Prophet
- Statsmodels
- HTML
- CSS
- JavaScript

## 📁 Project Structure

```text
InsightLens/
│
├── data/
│   └── generate_data.py
│
├── models/
│   ├── arima_model.py
│   ├── prophet_model.py
│   ├── xgboost_model.py
│   └── lstm_model.py
│
├── outputs/
│   └── forecast_data.csv
│
├── api.py
├── dashboard.html
├── feature_engineering.py
├── model_selector.py
└── README.md

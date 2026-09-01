import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from pathlib import Path


def create_features(df):
    df = df.copy()

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    return df


def train_xgboost(data_path, forecast_days=30):
    # Load data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Create features
    df = create_features(df)

    features = [
        "day",
        "month",
        "day_of_week",
        "day_of_year"
    ]

    X = df[features]
    y = df["sales"]

    # Create model
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )

    # Train model
    model.fit(X, y)

    # Create future dates
    last_date = df["date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    future_df = pd.DataFrame({
        "date": future_dates
    })

    future_df = create_features(future_df)

    # Predict
    predictions = model.predict(future_df[features])

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": np.round(predictions, 2)
    })

    return forecast_df


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "outputs" / "forecast_data.csv"

    forecast = train_xgboost(data_path)

    print("XGBoost forecast generated successfully!")
    print(forecast.head())
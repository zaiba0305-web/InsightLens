import pandas as pd
from prophet import Prophet
from pathlib import Path


def train_prophet(data_path, forecast_days=30):
    # Load data
    df = pd.read_csv(data_path)

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Prophet requires columns named ds and y
    prophet_df = df[["date", "sales"]].rename(
        columns={
            "date": "ds",
            "sales": "y"
        }
    )

    # Create and train model
    model = Prophet()
    model.fit(prophet_df)

    # Create future dates
    future = model.make_future_dataframe(
        periods=forecast_days,
        freq="D"
    )

    # Generate forecast
    forecast = model.predict(future)

    # Keep only future predictions
    forecast_df = forecast[
        ["ds", "yhat"]
    ].tail(forecast_days)

    # Rename columns
    forecast_df = forecast_df.rename(
        columns={
            "ds": "date",
            "yhat": "forecast"
        }
    )

    return forecast_df


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "outputs" / "forecast_data.csv"

    forecast = train_prophet(data_path)

    print("Prophet forecast generated successfully!")
    print(forecast.head())
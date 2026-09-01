import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path


def train_arima(data_path, forecast_days=30):
    # Load data
    df = pd.read_csv(data_path)

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Set date as index
    df = df.set_index("date")

    # Get sales data
    sales = df["sales"]

    # Train ARIMA model
    model = ARIMA(sales, order=(5, 1, 0))
    model_fit = model.fit()

    # Generate forecast
    forecast = model_fit.forecast(steps=forecast_days)

    # Create forecast dates
    last_date = df.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast": forecast.values
    })

    return forecast_df


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "outputs" / "forecast_data.csv"

    forecast = train_arima(data_path)

    print("ARIMA forecast generated successfully!")
    print(forecast.head())
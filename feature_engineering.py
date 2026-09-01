import pandas as pd


def create_features(data_path):
    """
    Load the forecasting dataset and create useful
    time-based features.
    """

    df = pd.read_csv(data_path)

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Time-based features
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Weekend indicator
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


if __name__ == "__main__":
    from pathlib import Path

    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "outputs" / "forecast_data.csv"

    data = create_features(data_path)

    print("Features created successfully!")
    print(data.head())
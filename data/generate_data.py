import pandas as pd
import numpy as np
from pathlib import Path

# Number of days of data
days = 365

# Create dates
dates = pd.date_range(
    start="2024-01-01",
    periods=days,
    freq="D"
)

# Make results reproducible
np.random.seed(42)

# Trend
trend = np.linspace(100, 180, days)

# Weekly seasonality
weekly_seasonality = 15 * np.sin(2 * np.pi * np.arange(days) / 7)

# Monthly seasonality
monthly_seasonality = 10 * np.sin(2 * np.pi * np.arange(days) / 30)

# Random noise
noise = np.random.normal(0, 8, days)

# Final sales values
sales = trend + weekly_seasonality + monthly_seasonality + noise

# Make sure values are positive
sales = np.maximum(sales, 1)

# Create dataframe
df = pd.DataFrame({
    "date": dates,
    "sales": np.round(sales, 2)
})

# Find project directory
project_dir = Path(__file__).resolve().parent.parent

# Create outputs/data location
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

# Save dataset
file_path = output_dir / "forecast_data.csv"
df.to_csv(file_path, index=False)

print("Dataset generated successfully!")
print(f"Saved to: {file_path}")
print(f"Total records: {len(df)}")
print(df.head())
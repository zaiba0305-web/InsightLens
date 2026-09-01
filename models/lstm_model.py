import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def train_lstm(data_path, forecast_days=30):
    # Load data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    values = df["sales"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    # Create sequences
    sequence_length = 30

    X = []
    y = []

    for i in range(sequence_length, len(scaled_values)):
        X.append(scaled_values[i - sequence_length:i])
        y.append(scaled_values[i])

    X = np.array(X)
    y = np.array(y)

    # Create LSTM model
    model = Sequential([
        LSTM(
            50,
            activation="tanh",
            input_shape=(sequence_length, 1)
        ),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    # Train
    model.fit(
        X,
        y,
        epochs=10,
        batch_size=16,
        verbose=0
    )

    # Start forecasting
    current_sequence = scaled_values[-sequence_length:].copy()
    predictions = []

    for _ in range(forecast_days):
        input_sequence = current_sequence.reshape(
            1,
            sequence_length,
            1
        )

        prediction = model.predict(
            input_sequence,
            verbose=0
        )[0][0]

        predictions.append(prediction)

        current_sequence = np.append(
            current_sequence[1:],
            [[prediction]],
            axis=0
        )

    # Convert predictions back to original scale
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    # Future dates
    last_date = df["date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": np.round(predictions, 2)
    })

    return forecast_df


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "outputs" / "forecast_data.csv"

    forecast = train_lstm(data_path)

    print("LSTM forecast generated successfully!")
    print(forecast.head())
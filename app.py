import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    layout="wide"
)

st.title("📈 Bitcoin Price Prediction Using GRU")

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "model_prediction_BTC_GRU.keras"
model = load_model(MODEL_PATH)

LOOKBACK = 60

# ===============================
# USER INPUT
# ===============================
days = st.number_input(
    "Predict how many days ahead?",
    min_value=1,
    max_value=365,
    value=30
)

# ===============================
# LOAD DATA
# ===============================
df = yf.download("BTC-USD", start="2015-01-01")
df = df[['Close']]
df.dropna(inplace=True)

# ===============================
# SCALING
# ===============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# ===============================
# LAST SEQUENCE
# ===============================
last_sequence = scaled_data[-LOOKBACK:]
current_input = last_sequence.reshape(1, LOOKBACK, 1)

predictions = []

# ===============================
# PREDICTION LOOP
# ===============================
for _ in range(days):
    next_pred = model.predict(current_input, verbose=0)
    predictions.append(next_pred[0, 0])

    current_input = np.append(
        current_input[:, 1:, :],
        next_pred.reshape(1, 1, 1),
        axis=1
    )

# ===============================
# INVERSE TRANSFORM
# ===============================
predictions = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
)

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=days
)

prediction_df = pd.DataFrame(
    predictions,
    index=future_dates,
    columns=["Predicted Price"]
)

# ===============================
# LIMIT HISTORICAL DATA (1 YEAR ONLY)
# ===============================
history_days = 365
df_plot = df[df.index >= df.index[-1] - pd.Timedelta(days=history_days)]

# ===============================
# PLOT
# ===============================
st.subheader("📊 Price Prediction Chart (Last 1 Year + Forecast)")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df_plot.index,
    df_plot['Close'],
    label="Historical Price",
    linewidth=2
)

ax.plot(
    prediction_df.index,
    prediction_df["Predicted Price"],
    label="Prediction",
    color="red",
    linewidth=2
)

# vertical line (today)
ax.axvline(
    x=df.index[-1],
    color="gray",
    linestyle="--",
    alpha=0.6,
    label="Last Actual Price"
)

ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# ===============================
# TABLE
# ===============================
st.subheader("📋 Prediction Table")
st.dataframe(
    prediction_df.style.format("${:,.2f}")
)

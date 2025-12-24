import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Cryptocurrency Price Prediction (GRU)",
    layout="wide"
)

# =====================
# Load Model (cache)
# =====================
@st.cache_resource
def load_gru_model():
    return load_model("model_prediction_BTC_GRU.keras")

model = load_gru_model()

# =====================
# Sidebar Input
# =====================
st.sidebar.title("⚙️ Prediction Settings")

ticker = st.sidebar.text_input(
    "Yahoo Finance Ticker",
    value="BTC-USD"
)

forecast_days = st.sidebar.number_input(
    "Number of days to predict",
    min_value=1,
    max_value=1000,
    value=30,
    step=1
)

if forecast_days > 365:
    st.sidebar.warning(
        "⚠️ Long-term predictions may accumulate higher errors"
    )

# =====================
# Load Historical Data
# =====================
@st.cache_data
def load_data(ticker):
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    return yf.download(ticker, start=start, end=end)

df = load_data(ticker)

# =====================
# Title
# =====================
st.title("📈 Cryptocurrency Price Prediction using GRU")
st.caption("Historical data from Yahoo Finance & deep learning-based forecasting")

if df.empty:
    st.error("❌ No data found. Please check the ticker symbol.")
    st.stop()

# =====================
# Historical Price Plot
# =====================
st.subheader("📊 Historical Closing Prices")

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df.index, df["Close"], label="Closing Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

# =====================
# Data Preparation
# =====================
WINDOW_SIZE = 100

close_prices = df[["Close"]]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_prices)

last_window = scaled_close[-WINDOW_SIZE:]
last_window = last_window.reshape(1, WINDOW_SIZE, 1)

# =====================
# Future Prediction (Recursive)
# =====================
future_predictions = []

for _ in range(forecast_days):
    next_pred = model.predict(last_window, verbose=0)
    future_predictions.append(next_pred[0, 0])

    last_window = np.append(
        last_window[:, 1:, :],
        next_pred.reshape(1, 1, 1),
        axis=1
    )

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
).flatten()

future_dates = [
    df.index[-1] + timedelta(days=i)
    for i in range(1, forecast_days + 1)
]

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close Price": future_predictions
})

# =====================
# Prediction Plot
# =====================
st.subheader("🔮 Future Price Forecast")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(
    future_df["Date"],
    future_df["Predicted Close Price"],
    marker="o"
)
ax2.set_xlabel("Date")
ax2.set_ylabel("Predicted Price")
ax2.grid(alpha=0.3)
st.pyplot(fig2)

# =====================
# Prediction Table
# =====================
st.subheader("📄 Forecast Table")
st.dataframe(future_df)

st.success("✅ Prediction completed successfully")

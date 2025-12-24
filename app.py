import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="BTC Price Prediction", layout="wide")

# =====================
# Load Model (cache)
# =====================
@st.cache_resource
def load_gru():
    return load_model("model_prediction_BTC_GRU.keras")

model = load_gru()

# =====================
# Sidebar Input
# =====================
st.sidebar.title("⚙️Setting")

stock = st.sidebar.text_input("Ticker Yahoo Finance", "BTC-USD")
no_of_days = st.sidebar.slider("Prediksi berapa hari ke depan?", 1, 30, 10)

# =====================
# Load Data
# =====================
@st.cache_data
def load_data(ticker):
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    df = yf.download(ticker, start=start, end=end)
    return df

stock_data = load_data(stock)

st.title(f"📈 {stock} Price Prediction")

if stock_data.empty:
    st.error("Ticker tidak valid atau data kosong")
    st.stop()

# =====================
# Plot 1: Closing Price
# =====================
st.subheader("📊 Harga Penutupan Historis")

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(stock_data['Close'], label='Close Price')
ax1.set_xlabel("Tanggal")
ax1.set_ylabel("Harga")
ax1.legend()
st.pyplot(fig1)

# =====================
# Data Preparation
# =====================
splitting_len = int(len(stock_data) * 0.9)
x_test = stock_data[['Close']][splitting_len:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data = np.array(x_data)
y_data = np.array(y_data)

# =====================
# Test Prediction
# =====================
predictions = model.predict(x_data, verbose=0)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    {
        "Original": inv_y_test.flatten(),
        "Prediction": inv_predictions.flatten(),
    },
    index=x_test.index[100:]
)

# =====================
# Plot 2: Original vs Prediction
# =====================
st.subheader("🔍 Original vs Prediksi (Test Data)")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(plotting_data["Original"], label="Original")
ax2.plot(plotting_data["Prediction"], label="Prediction", linestyle="--")
ax2.legend()
ax2.set_xlabel("Tanggal")
ax2.set_ylabel("Harga")
st.pyplot(fig2)

st.dataframe(plotting_data.tail())

# =====================
# Future Prediction (Recursive)
# =====================
st.subheader("🔮 Prediksi Harga Masa Depan")

last_100 = stock_data[['Close']].tail(100)
last_100_scaled = scaler.transform(last_100)
last_100_scaled = last_100_scaled.reshape(1, -1, 1)

future_predictions = []

for _ in range(no_of_days):
    next_day = model.predict(last_100_scaled, verbose=0)
    future_predictions.append(scaler.inverse_transform(next_day)[0, 0])

    last_100_scaled = np.append(
        last_100_scaled[:, 1:, :],
        next_day.reshape(1, 1, 1),
        axis=1
    )

future_dates = [
    stock_data.index[-1] + timedelta(days=i)
    for i in range(1, no_of_days + 1)
]

future_df = pd.DataFrame(
    {
        "Date": future_dates,
        "Predicted Close Price": future_predictions
    }
)

# =====================
# Plot 3: Future Prediction
# =====================
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(future_df["Date"], future_df["Predicted Close Price"], marker='o')
ax3.set_xlabel("Tanggal")
ax3.set_ylabel("Predicted Price")
ax3.grid(alpha=0.3)
st.pyplot(fig3)

st.dataframe(future_df)

st.success("🚀 Prediksi selesai")


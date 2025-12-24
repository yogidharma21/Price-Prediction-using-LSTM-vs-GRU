import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")
st.title("📈 Bitcoin Price Prediction Using GRU")

MODEL_PATH = "model_prediction_BTC_GRU.keras"
LOOKBACK = 30

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)

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

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# ===============================
# PREPARE LAST SEQUENCE
# ===============================
last_sequence = scaled_data[-LOOKBACK:]
current_input = last_sequence.reshape(1, LOOKBACK, 1)

predictions = []

# ===============================
# PREDICTION LOOP
# ===============================
for _ in range(days):
    pred = model.predict(current_input, verbose=0)
    predictions.append(pred[0, 0])

    current_input = np.append(
        current_input[:, 1:, :],
        pred.reshape(1, 1, 1),
        axis=1
    )

# ===============================
# INVERSE SCALE
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
# PLOT
# ===============================
st.subheader("📊 Price Prediction Chart")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df['Close'], label="Historical Price")
ax.plot(prediction_df.index, prediction_df["Predicted Price"], label="Prediction", color="red")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
st.pyplot(fig)

# ===============================
# TABLE
# ===============================
st.subheader("📋 Prediction Table")
st.dataframe(prediction_df.style.format("${:,.2f}"))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model, scaler, and dataset
model = joblib.load("spoilage_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("dataset.csv")

st.title("🍎 Fruit Spoilage Prediction Dashboard")

st.write("This dashboard predicts spoilage risk based on storage conditions.")

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Interactive Widgets
# -----------------------------
st.subheader("Adjust Conditions")

room_temp = st.slider("Room Temperature (°C)", float(df.room_temperature.min()), float(df.room_temperature.max()), 3.0)
cooling = st.slider("Cooling Intensity", 0.0, 1.0, 0.5)
storage_days = st.slider("Storage Days", 0, int(df.storage_days.max()), 5)
co2 = st.slider("CO₂ Level", float(df.co2.min()), float(df.co2.max()), 0.5)
firmness = st.slider("Firmness", float(df.firmness.min()), float(df.firmness.max()), 7.0)
electricity_price = st.slider("Electricity Price (€/kWh)", float(df.electricity_price.min()), float(df.electricity_price.max()), 0.25)

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Predicted Spoilage Risk")

input_data = pd.DataFrame([{
    "room_temperature": room_temp,
    "cooling_intensity": cooling,
    "electricity_price": electricity_price,
    "co2": co2,
    "firmness": firmness,
    "storage_days": storage_days
}])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

if prediction == 1:
    st.error(f"⚠️ Spoilage Likely (Probability: {probability:.2f})")
else:
    st.success(f"✔️ Spoilage Unlikely (Probability: {probability:.2f})")

# -----------------------------
# Plots
# -----------------------------
st.subheader("Room Temperature Over Time")
fig, ax = plt.subplots()
ax.plot(df["time"], df["room_temperature"])
ax.set_xlabel("Time")
ax.set_ylabel("Room Temperature (°C)")
st.pyplot(fig)

st.subheader("Electricity Price Over Time")
fig2, ax2 = plt.subplots()
ax2.plot(df["time"], df["electricity_price"], color="orange")
ax2.set_xlabel("Time")
ax2.set_ylabel("Electricity Price (€/kWh)")
st.pyplot(fig2)

st.subheader("Cooling Intensity Over Time")
fig3, ax3 = plt.subplots()
ax3.plot(df["time"], df["cooling_intensity"], color="green")
ax3.set_xlabel("Time")
ax3.set_ylabel("Cooling Intensity")
st.pyplot(fig3)
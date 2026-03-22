## Overview

This project simulates a cold‑storage environment for fruits and predicts spoilage risk using a machine‑learning model. It combines synthetic data generation, real weather API data, a custom spoilage formula, and an interactive Streamlit dashboard. The goal is to demonstrate how data science can support bioeconomy, food storage optimization, and energy‑aware decision‑making.

## 1. Data Generation Pipeline

This project uses a hybrid dataset created from:

# A. Real Weather API Data

We retrieved real hourly weather data (temperature, humidity, etc.) using a weather API. This provides realistic environmental variation over time.

# B. Synthetic Storage Variables

To simulate a real cold‑storage facility, we generated additional variables:

| Variable  | Description | 

| cooling_intensity | Random fluctuations between 0–1 to mimic compressor cycles | 

| electricity_price | Synthetic price curve with daily peaks and noise | 

| co2  | Random CO₂ levels representing respiration and ventilation | 

| firmness | Gradual decline over time with random noise |

| storage_days  | Counter representing how long the fruit has been stored | 

These variables allow us to model spoilage even without a real industrial dataset.

## 2. Spoilage Formula (Custom Linear Model)

To label the dataset (spoilage vs. no spoilage), we created a linear spoilage risk function inspired by postharvest biology:

spoilage_score= 0.4 x room_temperature + 0.3 x storage_days + 0.2 x co2 - 0.3 x cooling_intensity - 0.2 x firmness

Then we applied a threshold:

spoilage= 1 (if spoilage_score > 5)

spoilage = 0 otherwise

Why this formula?

- Higher temperature, CO₂, and storage duration accelerate spoilage.
- Higher cooling intensity and firmness reduce spoilage.
- Coefficients were chosen to create realistic class separation for ML training.

This approach is common in simulation‑based research when real spoilage labels are unavailable.

## 3. Machine Learning Models
We trained two models:

# A. Logistic Regression
- Accuracy: ~94%
- Performs well due to linear separability of the synthetic formula.

# B. Random Forest
- Accuracy: ~91%
- Provides feature importance, which we visualize in the dashboard.

# Feature Importance (Random Forest)
Typical ranking:
- Cooling intensity
- Electricity price
- Room temperature
- CO₂
- Firmness
- Storage days
  
This gives insight into which factors most influence spoilage in our simulation.

## 4. Streamlit Dashboard
The dashboard allows users to:

 # Adjust storage conditions
- Room temperature
- Cooling intensity
- CO₂
- Firmness
- Storage days
- Electricity price

 # Get real‑time spoilage predictions
The model outputs:
- Spoilage (Yes/No)
- Probability score

 # View time‑series plots
- Room temperature
- Electricity price
- Cooling intensity

 # Explore the dataset
 
A preview of the first rows is included.

## 5. Project Structure

fruit-spoilage-dashboard/

│

├── dashboard.py              # Streamlit app

├── spoilage_model.pkl        # Trained Random Forest model

├── scaler.pkl                # Scaler for preprocessing

├── dataset.csv               # Final dataset used in the dashboard

├── Spoilage_Model.ipynb      # Full notebook with data generation + training

└── requirements.txt          # Dependencies for deployment


# 6. Deployment
The dashboard is deployed using Streamlit Cloud.

Steps:
- Push all files to GitHub
- Go to https://share.streamlit.io
- Select your repo
- Choose dashboard.py as the entry point
- Deploy
Streamlit automatically installs dependencies and hosts the app.

## 7. Purpose of the Project

This project demonstrates:

- Applied data science for sustainability
- Simulation modeling when real data is limited
- Machine learning for spoilage prediction
- Dashboard design for decision support
- Integration of API data with synthetic variables

It is designed for academic evaluation, portfolio presentation, and internship/thesis applications in bioeconomy, data science, and energy systems.

## 8. Author
Lovepreet Kapila
M.Sc. Bioeconomy (Data Science & AI)
University of Hohenheim, Stuttgart

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and feature names
model = joblib.load("linear_regression_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Define original categorical options (replace with actual unique values from your dataset)
model_options = ['Fiesta', 'Focus', 'Mondeo']           # replace with actual models
transmission_options = ['Manual', 'Automatic']         # replace with actual transmissions
fuelType_options = ['Petrol', 'Diesel', 'Hybrid']      # replace with actual fuel types

# App title
st.title("Ford Car Price Prediction")
st.write("Fill in the car details to predict its price:")

# User inputs
st.header("Car Specifications")

# Categorical features
car_model = st.selectbox("Select Model", model_options)
transmission = st.selectbox("Select Transmission", transmission_options)
fuel_type = st.selectbox("Select Fuel Type", fuelType_options)

# Numeric features
year = st.slider("Year", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Mileage (in miles)", min_value=0, value=50000)
tax = st.number_input("Tax (£)", min_value=0, value=150)
mpg = st.number_input("MPG", min_value=0.0, value=40.0)
engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=2.0)

# Convert inputs to a DataFrame in the same format as training
input_dict = {
    'year': year,
    'mileage': mileage,
    'tax': tax,
    'mpg': mpg,
    'engineSize': engine_size,
    'model_' + car_model: 1,
    'transmission_' + transmission: 1,
    'fuelType_' + fuel_type: 1
}

# Initialize all features with 0, then update with user input
input_df = pd.DataFrame(np.zeros(len(feature_names)).reshape(1, -1), columns=feature_names)
for col, val in input_dict.items():
    if col in input_df.columns:
        input_df[col] = val

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Car Price: £{prediction:,.2f}")



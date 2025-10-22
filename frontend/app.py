# frontend/app.py
import streamlit as st
import requests
import numpy as np

st.set_page_config(page_title="Energy Efficiency Prediction", layout="centered")

st.title("🏠 Energy Efficiency Prediction")
st.write("Enter building parameters to predict heating load.")

# Example input fields (adjust to match your dataset)
relative_compactness = st.number_input("Relative Compactness", 0.5, 1.0, 0.8)
surface_area = st.number_input("Surface Area", 400.0, 900.0, 600.0)
wall_area = st.number_input("Wall Area", 200.0, 500.0, 300.0)
roof_area = st.number_input("Roof Area", 100.0, 400.0, 200.0)
overall_height = st.number_input("Overall Height", 2.5, 7.0, 3.5)
orientation = st.number_input("Orientation (1–4)", 1, 4, 2)
glazing_area = st.number_input("Glazing Area", 0.0, 0.4, 0.1)
glazing_area_dist = st.number_input("Glazing Area Distribution (1–5)", 1, 5, 5)

if st.button("Predict Heating Load"):
    data = {
        "features": [relative_compactness, surface_area, wall_area, roof_area,
                     overall_height, orientation, glazing_area, glazing_area_dist]
    }
    try:
        api_url = "https://energy-efficiency-api.onrender.com/predict"

        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            result = response.json()['predicted_heating_load']
            st.success(f"🔥 Predicted Heating Load: {result:.2f}")
        else:
            st.error("Prediction failed.")
    except Exception as e:
        st.error(f"Error: {e}")

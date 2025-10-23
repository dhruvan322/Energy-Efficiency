# frontend/app.py
import streamlit as st
import requests, json

st.set_page_config(page_title="Energy Efficiency Predictor (multi-model)", layout="centered")
st.title("🏠 Energy Efficiency Predictor (multiple algorithms)")

st.markdown("Enter building features and choose algorithms to get numeric (regression) and categorical (Low/Medium/High) predictions for Heating & Cooling loads.")

# Inputs
rc = st.number_input("Relative Compactness", 0.0, 2.0, 0.82, format="%.3f")
sa = st.number_input("Surface Area", 0.0, 2000.0, 550.0, format="%.2f")
wa = st.number_input("Wall Area", 0.0, 2000.0, 300.0, format="%.2f")
ra = st.number_input("Roof Area", 0.0, 2000.0, 220.0, format="%.2f")
oh = st.number_input("Overall Height", 0.0, 20.0, 3.5, format="%.2f")
ornt = st.number_input("Orientation (1-4)", 1, 4, 2, step=1)
ga = st.number_input("Glazing Area", 0.0, 1.0, 0.1, format="%.3f")
gad = st.number_input("Glazing Area Distribution (1-5)", 1, 5, 5, step=1)

# Backend URL
api_default = "http://127.0.0.1:5000"
base_url = st.text_input("Backend base URL (no trailing slash)", api_default)

# fetch available models from backend
if st.button("Refresh available models"):
    try:
        r = requests.get(base_url + "/")
        r.raise_for_status()
        models = r.json()
        st.session_state['models'] = models
        st.success("Models refreshed")
    except Exception as e:
        st.error(f"Could not fetch models: {e}")
        st.session_state['models'] = {}

models = st.session_state.get('models', None)
reg_options = ["LinearRegression","KNN","DecisionTree","AdaBoost","RandomForest","GradientBoost"]
clf_options = ["LogisticRegression","KNN","DecisionTree","AdaBoost","RandomForest","GradientBoost"]

reg_choice = st.selectbox("Regression algorithm (choose for both heating & cooling)", reg_options)
clf_choice = st.selectbox("Classification algorithm (choose for both heating & cooling)", clf_options)
show_all = st.checkbox("Return outputs from all saved models (ignore selected algorithm)")

# Predict button
if st.button("Predict"):
    api_url = base_url + "/predict"
    payload = {
        "features": [rc, sa, wa, ra, oh, ornt, ga, gad],
        "reg_model": reg_choice,
        "clf_model": clf_choice,
        "return_all": bool(show_all)
    }
    try:
        r = requests.post(api_url, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        st.write("### Input features")
        st.json(data.get('input', payload['features']))

        if show_all:
            st.write("### Regression outputs (heating)")
            st.json(data['results']['heating_regression'])
            st.write("### Regression outputs (cooling)")
            st.json(data['results']['cooling_regression'])
            st.write("### Classification outputs (heating)")
            st.json(data['results']['heating_classification'])
            st.write("### Classification outputs (cooling)")
            st.json(data['results']['cooling_classification'])
        else:
            st.write("### Heating")
            st.write(f"Regression model: **{data['results']['heating']['regression_model']}**")
            st.write(f"Predicted Heating Load: **{data['results']['heating']['regression_value']}**")
            st.write(f"Classification model: **{data['results']['heating']['classification_model']}**")
            st.write(f"Predicted class: **{data['results']['heating']['class_label']}**")

            st.write("### Cooling")
            st.write(f"Regression model: **{data['results']['cooling']['regression_model']}**")
            st.write(f"Predicted Cooling Load: **{data['results']['cooling']['regression_value']}**")
            st.write(f"Classification model: **{data['results']['cooling']['classification_model']}**")
            st.write(f"Predicted class: **{data['results']['cooling']['class_label']}**")
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# Show training metrics
if st.button("Show training metrics (from backend)"):
    try:
        r = requests.get(base_url + "/metrics", timeout=10)
        r.raise_for_status()
        metrics = r.json()
        st.write("### Training metrics (regression)")
        st.json(metrics.get('regression', {}))
        st.write("### Training metrics (classification)")
        st.json(metrics.get('classification', {}))
    except Exception as e:
        st.error(f"Could not fetch metrics: {e}")

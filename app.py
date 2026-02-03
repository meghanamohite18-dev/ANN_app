import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="Pile Settlement Prediction using ANN",
    layout="centered"
)

# Title
st.title("Clayey Soil Pile Settlement Prediction (ANN)")
st.write("Primary consolidation settlement based on ANN model")

# Load model & scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("clay_pile_settlement_ann.h5", compile=False)
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_model_and_scaler()

# Input section
st.subheader("Input Parameters")

I = st.number_input("Depth below Ground Level, I (m)", min_value=0.0, value=15.0)
L = st.number_input("Pile Length, L (m)", min_value=0.0, value=10.0)
LL = st.number_input("Liquid Limit, LL (%)", min_value=0.0, value=45.0)
e = st.number_input("Void Ratio, e", min_value=0.0, value=0.8)
gamma = st.number_input("Soil Unit Weight, γ (kN/m³)", min_value=0.0, value=18.0)
Q = st.number_input("Load on Pile, Q (kN)", min_value=0.0, value=500.0)
B = st.number_input("Pile Cap Width, B (m)", min_value=0.0, value=3.0)

# Prediction
if st.button("Predict Settlement"):
    input_df = pd.DataFrame(
        [[I, L, LL, e, gamma, Q, B]],
        columns=['I', 'L', 'LL', 'e', 'gamma', 'Q', 'B']
    )

    input_scaled = scaler.transform(input_df)
    settlement = model.predict(input_scaled)[0][0]

    st.success(f"Predicted Consolidation Settlement = {settlement:.4f} m")

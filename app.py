import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

st.title("Pile Settlement Prediction using ANN")
st.write("Enter the parameters below to get predicted settlement.")

# Input fields
I = st.number_input("I (Depth below GL)", value=5.0)
L = st.number_input("L (Length of structure)", value=10.0)
LL = st.number_input("LL (Liquid Limit)", value=30.0)
e0 = st.number_input("e0 (Initial void ratio)", value=0.8)
gamma = st.number_input("γ (Unit weight)", value=18.0)
Q = st.number_input("Q (Total load value)", value=100.0)
B = st.number_input("B (Width of pile group)", value=2.0)

# Load ANN model
model = tf.keras.models.load_model("model.h5", compile=False)

# Manually define the scaler (no joblib needed)
scaler = StandardScaler()
scaler.mean_ = np.array([5.0, 10.0, 30.0, 0.8, 18.0, 100.0, 2.0])
scaler.scale_ = np.array([1.0, 1.5, 5.0, 0.05, 2.0, 30.0, 0.5])

# Predict button
if st.button("Predict Settlement"):
    x = np.array([[I, L, LL, e0, gamma, Q, B]])
    x_scaled = scaler.transform(x)
    y_pred = model.predict(x_scaled)
    st.success(f"Predicted Settlement: {y_pred[0][0]:.4f} mm")

import streamlit as st

PASSWORD = "App.py"   # change your password here

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        pwd = st.text_input("Enter Code", type="password")

        if pwd == PASSWORD:
            st.session_state.password_correct = True
        else:
            st.warning("Incorrect Password")

    return st.session_state.password_correct

if not check_password():
    st.stop()

import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Pile Settlement Prediction using ANN",
    layout="centered"
)

st.title("Clayey Soil Pile Settlement Prediction (ANN)")
st.write("Primary consolidation settlement prediction for **pile foundations only**")

# --------------------------------------------------
# Load model and scaler
# --------------------------------------------------
@st.cache_resource
def load_all():
    model = load_model("clay_pile_settlement_ann.h5", compile=False)
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_all()

# --------------------------------------------------
# Inputs
# --------------------------------------------------
st.subheader("Input Parameters")

I = st.number_input("Depth below Ground Level, I (m)", 0.0, 50.0, 15.0)
L = st.number_input("Pile Length, L (m)", 0.0, 50.0, 10.0)
LL = st.number_input("Liquid Limit, LL (%)", 0.0, 100.0, 45.0)
e = st.number_input("Void Ratio, e", 0.0, 2.0, 0.8)
gamma = st.number_input("Soil Unit Weight, γ (kN/m³)", 0.0, 30.0, 18.0)
Q = st.number_input("Load on Pile, Q (kN)", 0.0, 2000.0, 500.0)
B = st.number_input("Pile Cap/Group Width, B (m)", 0.0, 10.0, 3.0)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Settlement"):

    X = pd.DataFrame(
        [[I, L, LL, e, gamma, Q, B]],
        columns=['I','L','LL','e','gamma','Q','B']
    )

    X_scaled = scaler.transform(X)
    settlement_m = model.predict(X_scaled)[0][0]
    settlement_mm = settlement_m * 1000

    st.success(f"Predicted Settlement = {settlement_m:.4f} m")
    st.write(f"📏 **In millimetres:** {settlement_mm:.2f} mm")

    # --------------------------------------------------
    # Allowable settlement check (PILE ONLY)
    # --------------------------------------------------
    ALLOWABLE = 20  # mm

    st.subheader("Pile Settlement Evaluation")
    st.write("🔹 **Allowable pile settlement:** 20 mm (IS 2911)")

    if settlement_mm <= ALLOWABLE:
        st.success("🟢 Settlement is within allowable limits. DESIGN IS SAFE.")
    else:
        st.error("🔴 Settlement exceeds allowable limits. DESIGN IS NOT SAFE.")

        st.subheader("Recommended Remedial Measures")
        st.markdown("""
        • Increase pile length  
        • Increase number of piles  
        • Adopt pile–raft foundation  
        • Ground improvement techniques  
        • Reduce applied structural load  

        *Based on IS 2911, IS 8009, Bowles & Braja M. Das*
        """)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption(
    "Scope limited to pile foundations in clayey soil. "
    "Allowable settlement as per IS 2911, IS 8009 (Part 1), "
    "Bowles (Foundation Analysis & Design) and Braja M. Das."
)



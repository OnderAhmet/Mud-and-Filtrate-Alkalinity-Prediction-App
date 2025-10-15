import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle

# ---------------------------
# Basic setup
# ---------------------------
st.set_page_config(page_title="Mud & Filtrate Alkalinity", layout="wide")
st.title("Mud (Pm) and Filtrate (Pf & Mf) Prediction")

# ---------------------------
# Sidebar info (kept)
# ---------------------------
st.sidebar.subheader("About this App")
st.sidebar.write("""
This app is developed as part of the thesis titled **'Development of Data-Driven Models for Estimating Mud and Filtrate Alkalinity Using Machine Learning Applications'**.
The thesis is written by **Ahmet Önder**, under the supervision of **Dr. Burak Kulga** and co-advisor **Dr. Sercan Gul**.
""")

# ---------------------------
# Lazy model loading (Streamlit 1.13 compatible)
# ---------------------------
@st.cache(allow_output_mutation=True)
def load_models_and_scaler():
    mf_model = xgb.XGBRegressor()
    mf_model.load_model('mf_model.xgb')

    pm_model = xgb.XGBRegressor()
    pm_model.load_model('pm_model.xgb')

    with open('pf_model.pkl', 'rb') as f:
        pf_model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return mf_model, pm_model, pf_model, scaler

# ---------------------------
# Inputs
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    Mud_Weight = st.number_input("Mud Weight, ppg", min_value=0.0, value=0.0)
    Chlorides = st.number_input("Chlorides, mg/L", min_value=0.0, value=0.0)
    HTHP_Fluid_Loss = st.number_input("HTHP Fluid Loss, cc/30min", min_value=0.0, value=0.0)
    NaCl_SA = st.number_input("NaCl (SA), %vol", min_value=0.0, value=0.0)
    Low_Gravity_SA = st.number_input("Low Gravity (SA), %vol", min_value=0.0, value=0.0)
    R600 = st.number_input("R600", min_value=0.0, value=0.0)
    R200 = st.number_input("R200", min_value=0.0, value=0.0)
    R6 = st.number_input("R6", min_value=0.0, value=0.0)
    Average_SG_Solids_SA = st.number_input("Average SG Solids (SA)", min_value=0.0, value=0.0)

with col2:
    Yield_Point = st.number_input("Yield Point, lbf/100 sqft", min_value=0.0, value=0.0)
    Solids = st.number_input("Solids, %vol", min_value=0.0, value=0.0)
    pH = st.number_input("pH", min_value=0.0, value=0.0)
    KCl_SA = st.number_input("KCl (SA), %vol", min_value=0.0, value=0.0)
    Drill_Solids_SA = st.number_input("Drill Solids (SA), %vol", min_value=0.0, value=0.0)
    R300 = st.number_input("R300", min_value=0.0, value=0.0)
    R100 = st.number_input("R100", min_value=0.0, value=0.0)
    R3 = st.number_input("R3", min_value=0.0, value=0.0)

# ---------------------------
# Prediction
# ---------------------------
if st.button('Predict'):
    mf_model, pm_model, pf_model, scaler = load_models_and_scaler()

    X_input = np.array([[
        Mud_Weight, Yield_Point, Chlorides, Solids, HTHP_Fluid_Loss, pH, NaCl_SA, KCl_SA,
        Low_Gravity_SA, Drill_Solids_SA, R600, R300, R200, R100, R6, R3, Average_SG_Solids_SA
    ]])

    X_input_df = pd.DataFrame(X_input, columns=[
        'Mud Weight, ppg', 'Yield Point, lbf/100 sqft', 'Chlorides, mg/L',
        'Solids, %vol', 'HTHP Fluid Loss, cc/30min', 'pH', 'NaCl (SA), %vol',
        'KCl (SA), %vol', 'Low Gravity (SA), %vol', 'Drill Solids (SA), %vol',
        'R600', 'R300', 'R200', 'R100', 'R6', 'R3', 'Average SG Solids (SA)'
    ])

    X_scaled = scaler.transform(X_input_df)
    Mf_pred = mf_model.predict(X_scaled)
    Pf_pred = pf_model.predict(np.array([[Mf_pred[0]]]))
    Pm_pred = pm_model.predict(np.array([[Mf_pred[0]]]))

    st.subheader(f"Predicted Mf: {Mf_pred[0]:.2f}")
    st.subheader(f"Predicted Pf: {Pf_pred[0]:.2f}")
    st.subheader(f"Predicted Pm: {Pm_pred[0]:.2f}")

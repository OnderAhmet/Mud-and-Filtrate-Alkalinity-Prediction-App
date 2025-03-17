import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Uygulama Başlığı
st.title("Mud (Pm) and Filtrate (Pf and Mf) Prediction")

# Bilgi Bölümü (Sol Tarafa)
st.sidebar.subheader("About this App")
st.sidebar.write("""
    This app is developed as part of the thesis titled **'Development of Data-Driven Models for Estimating Mud and Filtrate Alkalinity Using Machine Learning Applications'**.
    The thesis is written by **Ahmet Önder**, under the supervision of **Dr. Burak Kulga** and co-advisor **Dr. Sercan Gul**.
""")

# Model ve Scaler Yükleme
try:
    mf_model = xgb.XGBRegressor()
    mf_model.load_model('mf_model.xgb')
except Exception as e:
    st.error(f"mf_model yükleme hatası: {e}")
    raise

try:
    pm_model = xgb.XGBRegressor()
    pm_model.load_model('pm_model.xgb')
except Exception as e:
    st.error(f"pm_model yüklenmesi hatası: {e}")
    raise

try:
    with open('pf_model.pkl', 'rb') as f:
        pf_model = pickle.load(f)
except Exception as e:
    st.error(f"pf_model yükleme hatası: {e}")
    raise

try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"Scaler yüklenmesi hatası: {e}")
    raise

# Parametre Giriş Alanı
st.subheader("Input Parameters")

# İki Kolonlu Düzen
col1, col2 = st.columns(2)  # İki sütunlu düzen

# Sol Kolon (pH ve R300 birlikte)
with col1:
    Mud_Weight = st.number_input("Mud Weight, ppg", min_value=0.0)
    Chlorides = st.number_input("Chlorides, mg/L", min_value=0.0)
    HTHP_Fluid_Loss = st.number_input("HTHP Fluid Loss, cc/30min", min_value=0.0)
    NaCl_SA = st.number_input("NaCl (SA), %vol", min_value=0.0)
    Low_Gravity_SA = st.number_input("Low Gravity (SA), %vol", min_value=0.0)
    R600 = st.number_input("R600", min_value=0.0)
    R200 = st.number_input("R200", min_value=0.0)
    R6 = st.number_input("R6", min_value=0.0)
    Average_SG_Solids_SA = st.number_input("Average SG Solids (SA)", min_value=0.0)

# Sağ Kolon (R300, R200, R100 gibi parametreler)
with col2:
    Yield_Point = st.number_input("Yield Point, lbf/100 sqft", min_value=0.0)
    Solids = st.number_input("Solids, %vol", min_value=0.0)
    pH = st.number_input("pH", min_value=0.0)
    KCl_SA = st.number_input("KCl (SA), %vol", min_value=0.0)
    Drill_Solids_SA = st.number_input("Drill Solids (SA), %vol", min_value=0.0)
    R300 = st.number_input("R300", min_value=0.0)
    R100 = st.number_input("R100", min_value=0.0)
    R3 = st.number_input("R3", min_value=0.0)


# Tahmin Butonu
if st.button('Predict'):
    X_input = np.array([[Mud_Weight, Yield_Point, Chlorides, Solids, HTHP_Fluid_Loss, pH, NaCl_SA, KCl_SA,
                         Low_Gravity_SA, Drill_Solids_SA, R600, R300, R200, R100, R6, R3, Average_SG_Solids_SA]])

    X_input_df = pd.DataFrame(X_input, columns=['Mud Weight, ppg', 'Yield Point, lbf/100 sqft', 'Chlorides, mg/L',
                                                'Solids, %vol', 'HTHP Fluid Loss, cc/30min', 'pH', 'NaCl (SA), %vol',
                                                'KCl (SA), %vol', 'Low Gravity (SA), %vol', 'Drill Solids (SA), %vol',
                                                'R600', 'R300', 'R200', 'R100', 'R6', 'R3', 'Average SG Solids (SA)'])
    X_input_scaled = scaler.transform(X_input_df)

    Mf_pred = mf_model.predict(X_input_scaled)
    st.write(f"Predicted Mf: {Mf_pred[0]:.2f}")

    X_input_pf = np.array([[Mf_pred[0]]])
    Pf_pred = pf_model.predict(X_input_pf)
    st.write(f"Predicted Pf: {Pf_pred[0]:.2f}")

    X_input_pm = np.array([[Mf_pred[0]]])
    Pm_pred = pm_model.predict(X_input_pm)
    st.write(f"Predicted Pm: {Pm_pred[0]:.2f}")

# Excel Yükleme Alanı
st.subheader("Upload your input file in Excel format")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    for index, row in df.iterrows():
        X_input = np.array([[row['Mud Weight'], row['Yield Point'], row['Chlorides'], row['Solids'], row['HTHP Fluid Loss'],
                             row['pH'], row['NaCl (SA)'], row['KCl (SA)'], row['Low Gravity (SA)'], row['Drill Solids (SA)'],
                             row['R600'], row['R300'], row['R200'], row['R100'], row['R6'], row['R3'], row['Average SG Solids (SA)']]])
        X_input_df = pd.DataFrame(X_input, columns=['Mud Weight', 'Yield Point', 'Chlorides', 'Solids', 'HTHP Fluid Loss',
                                                    'pH', 'NaCl (%vol)', 'KCl (%vol)', 'Low Gravity (%vol)',
                                                    'Drill Solids (%vol)', 'R600', 'R300', 'R200', 'R100', 'R6',
                                                    'R3', 'Average SG Solids'])
        X_input_scaled = scaler.transform(X_input_df)

        Mf_pred = mf_model.predict(X_input_scaled)
        df.loc[index, 'Mf'] = Mf_pred[0]

        X_input_pf = np.array([[Mf_pred[0]]])
        Pf_pred = pf_model.predict(X_input_pf)
        df.loc[index, 'Pf'] = Pf_pred[0]

        Pm_pred = pm_model.predict(X_input_pf)
        df.loc[index, 'Pm'] = Pm_pred[0]

    output_filename = "prediction_results.xlsx"
    df.to_excel(output_filename, index=False)
    st.write(f"Results saved.: {output_filename}")
    st.download_button(label="Download your excel file.", data=output_filename)

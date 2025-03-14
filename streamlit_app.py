import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Pickle dosyalarını yükleme
with open('mf_model.pkl', 'rb') as f:
    mf_model = pickle.load(f)

with open('pf_model.pkl', 'rb') as f:
    pf_model = pickle.load(f)

with open('pm_model.pkl', 'rb') as f:
    pm_model = pickle.load(f)

# Başlık
st.title("Mud (Pm) and Filtrate (Pf and Mf) Prediction")

# 1. Yol: Kullanıcıdan manuel parametre girişi
st.subheader("Manuel Parametre Girişi")

Mud_Weight = st.number_input("Mud Weight (ppg)", min_value=0.0)
Yield_Point = st.number_input("Yield Point (lbf/100 sqft)", min_value=0.0)
Chlorides = st.number_input("Chlorides (mg/L)", min_value=0.0)
Solids = st.number_input("Solids (%vol)", min_value=0.0)
HTHP_Fluid_Loss = st.number_input("HTHP Fluid Loss (cc/30min)", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
NaCl_SA = st.number_input("NaCl (%vol)", min_value=0.0)
KCl_SA = st.number_input("KCl (%vol)", min_value=0.0)
Low_Gravity_SA = st.number_input("Low Gravity (%vol)", min_value=0.0)
Drill_Solids_SA = st.number_input("Drill Solids (%vol)", min_value=0.0)
Average_SG_Solids_SA = st.number_input("Average SG Solids", min_value=0.0)
R600 = st.number_input("R600", min_value=0.0)
R300 = st.number_input("R300", min_value=0.0)
R200 = st.number_input("R200", min_value=0.0)
R100 = st.number_input("R100", min_value=0.0)
R6 = st.number_input("R6", min_value=0.0)
R3 = st.number_input("R3", min_value=0.0)


# Mf tahmin etme
if st.button('Mf Tahmin Et'):
    X_input = np.array([[Mud_Weight, Yield_Point, Chlorides, Solids, HTHP_Fluid_Loss, pH, NaCl_SA, KCl_SA,
                         Low_Gravity_SA, Drill_Solids_SA, R600, R300, R200, R100, R6, R3, Average_SG_Solids_SA]])
    Mf_pred = mf_model.predict(X_input)
    st.write(f"Predicted Mf: {Mf_pred[0]}")

# Pf tahmin etme
if st.button('Pf Tahmin Et'):
    Mf_value = Mf_pred[0]  # Mf modelinden alınan tahmin
    X_input = np.array([[Mf_value]])
    Pf_pred = pf_model.predict(X_input)
    st.write(f"Predicted Pf: {Pf_pred[0]}")

# Pm tahmin etme
if st.button('Pm Tahmin Et'):
    Mf_value = Mf_pred[0]  # Mf modelinden alınan tahmin
    X_input = np.array([[Mf_value]])
    Pm_pred = pm_model.predict(X_input)
    st.write(f"Predicted Pm: {Pm_pred[0]}")

# 2. Yol: Excel dosyası yükleme
st.subheader("Excel Dosyası Yükle")
uploaded_file = st.file_uploader("Excel dosyanızı yükleyin", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # MF, PF ve PM tahminlerini yapma
    for index, row in df.iterrows():
        X_input = np.array(
            [[row['Mud Weight'], row['Yield Point'], row['Chlorides'], row['Solids'], row['HTHP Fluid Loss'],
              row['pH'], row['NaCl (SA)'], row['KCl (SA)'], row['Low Gravity (SA)'], row['Drill Solids (SA)'],
              row['R600'], row['R300'], row['R200'], row['R100'], row['R6'], row['R3'], row['Average SG Solids (SA)']]])
        Mf_pred = mf_model.predict(X_input)
        df.loc[index, 'Mf'] = Mf_pred[0]

        X_input_pf = np.array([[Mf_pred[0]]])
        Pf_pred = pf_model.predict(X_input_pf)
        df.loc[index, 'Pf'] = Pf_pred[0]

        Pm_pred = pm_model.predict(X_input_pf)
        df.loc[index, 'Pm'] = Pm_pred[0]

    # Sonuçları yeni bir Excel dosyasına kaydetme
    output_filename = "prediction_results.xlsx"
    df.to_excel(output_filename, index=False)
    st.write(f"Sonuçlar kaydedildi: {output_filename}")
    st.download_button(label="Excel dosyasını indir", data=output_filename)

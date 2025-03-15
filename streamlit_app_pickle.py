import streamlit as st
import xgboost as xgb  # xgboost'u yükledik
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# mf_model ve pm_model XGBoost kullanılarak .xgb formatında kaydedildiği için XGBoost ile yükleyin
try:
    # mf_model yüklenmesi
    mf_model = xgb.XGBRegressor()
    mf_model.load_model('mf_model.xgb')
except Exception as e:
    st.error(f"mf_model yükleme hatası: {e}")
    raise

try:
    # pm_model yüklenmesi
    pm_model = xgb.XGBRegressor()
    pm_model.load_model('pm_model.xgb')
except Exception as e:
    st.error(f"pm_model yüklenmesi hatası: {e}")
    raise

# pf_model için pickle ile Random Forest modelini yükleyin
try:
    with open('pf_model.pkl', 'rb') as f:
        pf_model = pickle.load(f)
except Exception as e:
    st.error(f"pf_model yükleme hatası: {e}")
    raise

try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))  # Eğitimde kullanılan scaler'ı yükle
except Exception as e:
    st.error(f"Scaler yüklenmesi hatası: {e}")
    raise

# Başlık
st.title("Mud (Pm) and Filtrate (Pf and Mf) Prediction")

# 1. Yol: Kullanıcıdan manuel parametre girişi
st.subheader("Input Parameters")

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
R600 = st.number_input("R600", min_value=0.0)
R300 = st.number_input("R300", min_value=0.0)
R200 = st.number_input("R200", min_value=0.0)
R100 = st.number_input("R100", min_value=0.0)
R6 = st.number_input("R6", min_value=0.0)
R3 = st.number_input("R3", min_value=0.0)
Average_SG_Solids_SA = st.number_input("Average SG Solids", min_value=0.0)

# `Mf_pred`, `Pf_pred`, ve `Pm_pred`'i global bir değişken olarak tutmak için
Mf_pred = None
Pf_pred = None
Pm_pred = None

# Tüm tahminleri tek bir butonla yapma
if st.button('Predict All'):
    # Kullanıcıdan gelen parametrelerle X_input oluşturun
    X_input = np.array([[Mud_Weight, Yield_Point, Chlorides, Solids, HTHP_Fluid_Loss, pH, NaCl_SA, KCl_SA,
                         Low_Gravity_SA, Drill_Solids_SA, R600, R300, R200, R100, R6, R3, Average_SG_Solids_SA]])

    # X_input'u pandas DataFrame'e dönüştürme
    X_input_df = pd.DataFrame(X_input, columns=['Mud Weight', 'Yield Point', 'Chlorides', 'Solids', 'HTHP Fluid Loss',
                                                'pH', 'NaCl (%vol)', 'KCl (%vol)', 'Low Gravity (%vol)',
                                                'Drill Solids (%vol)', 'R600', 'R300', 'R200', 'R100', 'R6',
                                                'R3', 'Average SG Solids'])

    # Veriyi ölçeklendirme
    X_input_scaled = scaler.transform(X_input_df)

    # Mf modelinden tahmin yapın
    Mf_pred = mf_model.predict(X_input_scaled)  # XGB'nin DMatrix formatı ile
    st.write(f"Predicted Mf: {round(Mf_pred[0], 2)}")  # Mf tahmini virgülden sonra 2 haneli

    # Mf tahmininden Pf'yi tahmin etme
    X_input_pf = np.array([[Mf_pred[0]]])
    Pf_pred = pf_model.predict(X_input_pf)  # XGB'nin DMatrix formatı ile
    st.write(f"Predicted Pf: {round(Pf_pred[0], 2)}")  # Pf tahmini virgülden sonra 2 haneli

    # Mf tahmininden Pm'yi tahmin etme
    X_input_pm = np.array([[Mf_pred[0]]])
    Pm_pred = pm_model.predict(X_input_pm)  # XGB'nin DMatrix formatı ile
    st.write(f"Predicted Pm: {round(Pm_pred[0], 2)}")  # Pm tahmini virgülden sonra 2 haneli

# 2. Yol: Excel dosyası yükleme
st.subheader("Upload your input file in Excel format")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # MF, PF ve PM tahminlerini yapma
    for index, row in df.iterrows():
        X_input = np.array(
            [[row['Mud Weight'], row['Yield Point'], row['Chlorides'], row['Solids'], row['HTHP Fluid Loss'],
              row['pH'], row['NaCl (SA)'], row['KCl (SA)'], row['Low Gravity (SA)'], row['Drill Solids (SA)'],
              row['R600'], row['R300'], row['R200'], row['R100'], row['R6'], row['R3'], row['Average SG Solids (SA)']]])

        # Veriyi pandas DataFrame'e dönüştürme
        X_input_df = pd.DataFrame(X_input, columns=['Mud Weight', 'Yield Point', 'Chlorides', 'Solids', 'HTHP Fluid Loss',
                                                    'pH', 'NaCl (%vol)', 'KCl (%vol)', 'Low Gravity (%vol)',
                                                    'Drill Solids (%vol)', 'R600', 'R300', 'R200', 'R100', 'R6',
                                                    'R3', 'Average SG Solids'])

        # Veriyi ölçeklendirme
        X_input_scaled = scaler.transform(X_input_df)

        Mf_pred = mf_model.predict(X_input_scaled)  # DMatrix kullanmaya gerek yok, doğrudan NumPy array ile çalışır
        df.loc[index, 'Mf'] = Mf_pred[0]

        X_input_pf = np.array([[Mf_pred[0]]])
        Pf_pred = pf_model.predict(X_input_pf)  # XGB'nin DMatrix formatı ile
        df.loc[index, 'Pf'] = Pf_pred[0]

        Pm_pred = pm_model.predict(X_input_pf)  # XGB'nin DMatrix formatı ile
        df.loc[index, 'Pm'] = Pm_pred[0]

    # Sonuçları yeni bir Excel dosyasına kaydetme
    output_filename = "prediction_results.xlsx"
    df.to_excel(output_filename, index=False)
    st.write(f"Results saved.: {output_filename}")
    st.download_button(label="Download your excel file.", data=output_filename)

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# MF modelini eğitme ve pickle dosyasına kaydetme
file_path_mf = 'C:/Users/ITU/Desktop/MUD_Project/ML_data_cleaned2.xlsx'
df_mf = pd.read_excel(file_path_mf)

# Özellikler ve hedef değişken
numeric_columns = ['Mud Weight, ppg', 'Yield Point, lbf/100 sqft', 'Chlorides, mg/L',
                   'Solids, %vol', 'HTHP Fluid Loss, cc/30min', 'pH',
                   'NaCl (SA), %vol', 'KCl (SA), %vol', 'Low Gravity (SA), %vol',
                   'Drill Solids (SA), %vol', 'R600', 'R300', 'R200', 'R100', 'R6', 'R3', 'Average SG Solids (SA)']
df_mf[numeric_columns] = df_mf[numeric_columns].apply(pd.to_numeric, errors='coerce')
df_mf = df_mf.dropna()

X_mf = df_mf[numeric_columns]
y_mf = df_mf['Mf'].values

preprocessor_mf = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_columns)])
X_mf_transformed = preprocessor_mf.fit_transform(X_mf)

X_train_mf, X_test_mf, y_train_mf, y_test_mf = train_test_split(X_mf_transformed, y_mf, test_size=0.2, random_state=42)

xgb_mf = XGBRegressor(n_estimators=450, learning_rate=0.1, max_depth=5, reg_alpha=1, reg_lambda=10, tree_method='hist')
xgb_mf.fit(X_train_mf, y_train_mf)

xgb_mf.save_model('mf_model.json')

with open('mf_model.pkl', 'wb') as f:
    pickle.dump(xgb_mf, f)

# PF modelini eğitme ve pickle dosyasına kaydetme
file_path_pf = 'C:/Users/ITU/Desktop/MUD_Project/ML_data_cleaned_Pf_Prediction.xlsx'
df_pf = pd.read_excel(file_path_pf)

X_pf = df_pf[['Mf']].values
y_pf = df_pf['Pf'].values

X_train_pf, X_test_pf, y_train_pf, y_test_pf = train_test_split(X_pf, y_pf, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=250, max_depth=5, min_samples_split=10, min_samples_leaf=3, random_state=42)
rf_model.fit(X_train_pf, y_train_pf)

with open('pf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# PM modelini eğitme ve pickle dosyasına kaydetme
file_path_pm = 'C:/Users/ITU/Desktop/MUD_Project/ML_data_cleaned_Pm_Prediction.xlsx'
df_pm = pd.read_excel(file_path_pm)

X_pm = df_pm[['Mf']].values
y_pm = df_pm['Alkal Mud (Pm)'].values

X_train_pm, X_test_pm, y_train_pm, y_test_pm = train_test_split(X_pm, y_pm, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.05, random_state=42, tree_method='hist')
xgb_model.fit(X_train_pm, y_train_pm)

xgb_model.save_model('pm_model.json')

with open('pm_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("Modeller başarıyla kaydedildi.")

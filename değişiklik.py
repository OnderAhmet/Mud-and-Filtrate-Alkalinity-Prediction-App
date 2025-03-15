import xgboost as xgb

# Modelinizi yükleyin
model = xgb.XGBRegressor()
model.load_model('C:/Users/ITU/Desktop/MUD_Project/Mud-and-Filtrate-Alkalinity/mf_model.pkl')  # Modeli yükleyin

# Yeni sürümle tekrar kaydedin
model.save_model('new_model.xgb')  # Yeni model dosyasını kaydedin

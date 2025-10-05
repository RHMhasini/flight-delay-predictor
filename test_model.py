import pickle
import pandas as pd
import json

print("="*60)
print("TESTING ALL EXPORTS")
print("="*60)

# Test 1: Load XGBoost
print("\n[1/7] Loading XGBoost...")
with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
print("✓ XGBoost loaded")

# Test 2: Load Random Forest
print("\n[2/7] Loading Random Forest...")
with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
print("✓ Random Forest loaded")

# Test 3: Load Logistic Regression
print("\n[3/7] Loading Logistic Regression...")
with open('models/logistic_model.pkl', 'rb') as f:
    log_data = pickle.load(f)
    log_model = log_data['model']
    scaler = log_data['scaler']
print("✓ Logistic Regression + Scaler loaded")

# Test 4: Load Airlines
print("\n[4/7] Loading Airlines...")
airlines = pd.read_csv('data/airlines.csv')
print(f"✓ Loaded {len(airlines)} airlines")
print(airlines.head())

# Test 5: Load Airports
print("\n[5/7] Loading Airports...")
airports = pd.read_csv('data/airports.csv')
print(f"✓ Loaded {len(airports)} airports")
print(airports.head())

# Test 6: Load Feature Names
print("\n[6/7] Loading Feature Names...")
with open('data/feature_names.txt', 'r') as f:
    features = f.read().splitlines()
print(f"✓ Loaded {len(features)} features")
print(features)

# Test 7: Load Feature Info
print("\n[7/7] Loading Feature Info...")
with open('data/feature_info.json', 'r') as f:
    feature_info = json.load(f)
print(f"✓ Feature info loaded")
print(f"Distance range: {feature_info['distance_min']} - {feature_info['distance_max']}")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
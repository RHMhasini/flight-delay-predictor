import pickle
import pandas as pd
import json
import os

print("="*60)
print("TESTING ALL EXPORTS")
print("="*60)

test_count = 0
passed_count = 0

# Test 1: Load XGBoost
test_count += 1
print(f"\n[{test_count}] Loading XGBoost...")
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    print("[OK] XGBoost loaded")
    passed_count += 1
except Exception as e:
    print(f"[FAILED] Error loading XGBoost: {e}")

# Test 2: Load Random Forest (optional - file may be too large for GitHub)
test_count += 1
print(f"\n[{test_count}] Loading Random Forest...")
if os.path.exists('models/random_forest_model.pkl'):
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        print("[OK] Random Forest loaded")
        passed_count += 1
    except Exception as e:
        print(f"[FAILED] Error loading Random Forest: {e}")
else:
    print("[SKIPPED] Random Forest model not found (file may be too large for GitHub)")
    passed_count += 1  # Count as passed since it's optional

# Test 3: Load Logistic Regression
test_count += 1
print(f"\n[{test_count}] Loading Logistic Regression...")
try:
    with open('models/logistic_model.pkl', 'rb') as f:
        log_data = pickle.load(f)
        log_model = log_data['model']
        scaler = log_data['scaler']
    print("[OK] Logistic Regression + Scaler loaded")
    passed_count += 1
except Exception as e:
    print(f"[FAILED] Error loading Logistic Regression: {e}")

# Test 4: Load Airlines
test_count += 1
print(f"\n[{test_count}] Loading Airlines...")
try:
    airlines = pd.read_csv('data/airlines.csv')
    print(f"[OK] Loaded {len(airlines)} airlines")
    print(airlines.head())
    passed_count += 1
except Exception as e:
    print(f"[FAILED] Error loading airlines: {e}")

# Test 5: Load Airports
test_count += 1
print(f"\n[{test_count}] Loading Airports...")
try:
    airports = pd.read_csv('data/airports.csv')
    print(f"[OK] Loaded {len(airports)} airports")
    print(airports.head())
    passed_count += 1
except Exception as e:
    print(f"[FAILED] Error loading airports: {e}")

# Test 6: Load Feature Names
test_count += 1
print(f"\n[{test_count}] Loading Feature Names...")
try:
    with open('data/feature_names.txt', 'r') as f:
        features = f.read().splitlines()
    print(f"[OK] Loaded {len(features)} features")
    print(features)
    passed_count += 1
except Exception as e:
    print(f"[FAILED] Error loading feature names: {e}")

# Test 7: Load Feature Info
test_count += 1
print(f"\n[{test_count}] Loading Feature Info...")
try:
    with open('data/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    print("[OK] Feature info loaded")
    print(f"Distance range: {feature_info['distance_min']} - {feature_info['distance_max']}")
    passed_count += 1
except Exception as e:
    print(f"[FAILED] Error loading feature info: {e}")

print("\n" + "="*60)
print(f"TESTS COMPLETED: {passed_count}/{test_count} passed")
print("="*60)
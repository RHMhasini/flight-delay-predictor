import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime, time
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# Title
st.title("✈️ Flight Delay Prediction System")
st.markdown("Predict flight delays using Machine Learning models")
st.markdown("---")

# Load models and data
@st.cache_resource
def load_models():
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/logistic_model.pkl', 'rb') as f:
        log_data = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return xgb_model, log_data, label_encoders

@st.cache_data
def load_data():
    airlines = pd.read_csv('data/airlines.csv')
    airports = pd.read_csv('data/airports.csv')
    with open('data/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    return airlines, airports, feature_info

# Load everything
xgb_model, log_data, label_encoders = load_models()
airlines_df, airports_df, feature_info = load_data()

# Helper functions
def get_time_block(hour):
    """Convert hour to time block TEXT LABEL"""
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 22:
        return 'evening'
    else:
        return 'night'

def get_distance_category(distance):
    """Convert distance to category TEXT LABEL"""
    if distance < 500:
        return 'short'
    elif distance < 1000:
        return 'medium'
    else:
        return 'long'

def create_csv_export(flight_info, predictions):
    """Create CSV data for export"""
    data = {
        'Airline': [flight_info['airline']],
        'Flight_Number': [flight_info['flight_number']],
        'Origin': [flight_info['origin']],
        'Destination': [flight_info['destination']],
        'Date': [flight_info['date']],
        'Departure_Time': [flight_info['dep_time']],
        'Arrival_Time': [flight_info['arr_time']],
        'Duration_Minutes': [flight_info['duration']],
        'Distance_Miles': [flight_info['distance']]
    }
    
    # Add predictions
    for model_name, pred_info in predictions.items():
        data[f'{model_name}_Prediction'] = [pred_info['result']]
        data[f'{model_name}_Confidence'] = [f"{pred_info['confidence']:.1f}%"]
    
    return pd.DataFrame(data)

def preprocess_input(airline_code, flight_num, origin, dest, dep_time, arr_time, 
                     elapsed_time, distance, flight_date):
    """Convert user inputs to model features"""
    
    # Date features
    day_of_week = flight_date.weekday()  # 0=Monday, 6=Sunday
    month = flight_date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Time features
    dep_hour = dep_time.hour
    time_block = get_time_block(dep_hour)
    crs_dep_time = dep_hour + (dep_time.minute / 60)
    crs_arr_time = arr_time.hour + (arr_time.minute / 60)
    
    # Distance category
    distance_cat = get_distance_category(distance)
    
    # Create feature dictionary with TEXT LABELS (not encoded yet)
    features = {
        'AIRLINE_CODE': airline_code,
        'FL_NUMBER': flight_num,
        'ORIGIN': origin,
        'DEST': dest,
        'CRS_DEP_TIME': crs_dep_time,
        'CRS_ARR_TIME': crs_arr_time,
        'CRS_ELAPSED_TIME': elapsed_time,
        'DISTANCE': distance,
        'DAY_OF_WEEK': day_of_week,
        'MONTH': month,
        'IS_WEEKEND': is_weekend,
        'DEP_HOUR': dep_hour,
        'TIME_BLOCK': time_block,
        'DISTANCE_CAT': distance_cat
    }
    
    df = pd.DataFrame([features])
    
    # NOW encode the categorical columns using the label encoders
    for col in ['AIRLINE_CODE', 'ORIGIN', 'DEST', 'TIME_BLOCK', 'DISTANCE_CAT']:
        if col in label_encoders:
            df[col] = df[col].map(label_encoders[col])
    
    return df

# Sidebar - User Inputs
st.sidebar.header("Flight Information")

# Airline selection
airline_name = st.sidebar.selectbox(
    "Select Airline",
    airlines_df['AIRLINE_NAME'].values
)
airline_code = airlines_df[airlines_df['AIRLINE_NAME'] == airline_name]['AIRLINE_CODE'].values[0]

# Flight number
flight_number = st.sidebar.number_input("Flight Number", min_value=1, max_value=9999, value=1234)

# Date selection
flight_date = st.sidebar.date_input("Flight Date", datetime.now())

# Origin airport - sort alphabetically for easier selection
airports_sorted = airports_df.sort_values('AIRPORT_NAME')
origin_name = st.sidebar.selectbox(
    "Origin Airport",
    airports_sorted['AIRPORT_NAME'].values
)
origin_code = airports_df[airports_df['AIRPORT_NAME'] == origin_name]['AIRPORT_CODE'].values[0]

# Destination airport
dest_name = st.sidebar.selectbox(
    "Destination Airport",
    airports_sorted['AIRPORT_NAME'].values
)
dest_code = airports_df[airports_df['AIRPORT_NAME'] == dest_name]['AIRPORT_CODE'].values[0]

# Departure time
dep_time = st.sidebar.time_input("Departure Time", time(14, 30))

# Arrival time
arr_time = st.sidebar.time_input("Arrival Time", time(17, 0))

# Calculate elapsed time
elapsed_minutes = (arr_time.hour * 60 + arr_time.minute) - (dep_time.hour * 60 + dep_time.minute)
if elapsed_minutes < 0:
    elapsed_minutes += 24 * 60  # Handle overnight flights

# Validate flight duration
is_valid_flight = True
validation_message = ""

if elapsed_minutes < 30:
    is_valid_flight = False
    validation_message = "Flight duration is less than 30 minutes. Please check your departure and arrival times."
elif elapsed_minutes > 18 * 60:  # More than 18 hours
    is_valid_flight = False
    validation_message = "Flight duration exceeds 18 hours. This seems unrealistic for most flights. Please verify your times."

# Show validation error if needed
if not is_valid_flight:
    st.sidebar.error(f"❌ {validation_message}")
else:
    st.sidebar.success(f"✓ Flight duration: {elapsed_minutes // 60}h {elapsed_minutes % 60}m")

# Distance
distance = st.sidebar.number_input("Distance (miles)", min_value=0, max_value=5000, value=1000)

# Validate that origin and destination are different
if origin_code == dest_code:
    is_valid_flight = False
    st.sidebar.error("❌ Origin and destination airports cannot be the same.")

# Model selection (removed Random Forest)
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost", "Logistic Regression", "Both Models"]
)

# Predict button - disabled if validation fails
predict_button = st.sidebar.button(
    "🔮 Predict Delay", 
    type="primary",
    disabled=not is_valid_flight
)

# Main area
if predict_button:
    # Preprocess input
    input_df = preprocess_input(
        airline_code, flight_number, origin_code, dest_code,
        dep_time, arr_time, elapsed_minutes, distance, flight_date
    )
    
    # Display input summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Details")
        st.write(f"**Airline:** {airline_name}")
        st.write(f"**Flight:** {airline_code}{flight_number}")
        st.write(f"**Route:** {origin_name} → {dest_name}")
        st.write(f"**Distance:** {distance} miles")
    
    with col2:
        st.subheader("Schedule")
        st.write(f"**Date:** {flight_date}")
        st.write(f"**Departure:** {dep_time}")
        st.write(f"**Arrival:** {arr_time}")
        st.write(f"**Duration:** {elapsed_minutes // 60}h {elapsed_minutes % 60}m")
    
    st.markdown("---")
    
    # Store predictions for export
    all_predictions = {}
    
    # Make predictions
    if model_choice == "XGBoost" or model_choice == "Both Models":
        st.subheader("XGBoost Prediction")
        xgb_pred = xgb_model.predict(input_df)[0]
        xgb_proba = xgb_model.predict_proba(input_df)[0]
        xgb_result = "DELAYED" if xgb_pred == 1 else "ON-TIME"
        xgb_confidence = xgb_proba[1] if xgb_pred == 1 else xgb_proba[0]
        
        all_predictions['XGBoost'] = {
            'result': xgb_result,
            'confidence': xgb_confidence * 100
        }
        
        if xgb_result == "DELAYED":
            st.error(f"🔴 {xgb_result} (Confidence: {xgb_confidence*100:.1f}%)")
        else:
            st.success(f"🟢 {xgb_result} (Confidence: {xgb_confidence*100:.1f}%)")
        
        if model_choice != "Both Models":
            st.markdown("---")
    
    if model_choice == "Logistic Regression" or model_choice == "Both Models":
        st.subheader("Logistic Regression Prediction")
        input_scaled = log_data['scaler'].transform(input_df)
        log_pred = log_data['model'].predict(input_scaled)[0]
        log_proba = log_data['model'].predict_proba(input_scaled)[0]
        log_result = "DELAYED" if log_pred == 1 else "ON-TIME"
        log_confidence = log_proba[1] if log_pred == 1 else log_proba[0]
        
        all_predictions['Logistic_Regression'] = {
            'result': log_result,
            'confidence': log_confidence * 100
        }
        
        if log_result == "DELAYED":
            st.error(f"🔴 {log_result} (Confidence: {log_confidence*100:.1f}%)")
        else:
            st.success(f"🟢 {log_result} (Confidence: {log_confidence*100:.1f}%)")
    
    # Export section
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    # Prepare flight info
    flight_info = {
        'airline': airline_name,
        'flight_number': f"{airline_code}{flight_number}",
        'origin': origin_name,
        'destination': dest_name,
        'date': str(flight_date),
        'dep_time': str(dep_time),
        'arr_time': str(arr_time),
        'duration': elapsed_minutes,
        'distance': distance
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        csv_data = create_csv_export(flight_info, all_predictions)
        csv_buffer = BytesIO()
        csv_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="📊 Download as CSV",
            data=csv_buffer,
            file_name=f"flight_prediction_{airline_code}{flight_number}_{flight_date}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Text Export
        text_report = f"""FLIGHT DELAY PREDICTION REPORT
{'='*50}

FLIGHT INFORMATION:
  Airline: {airline_name}
  Flight: {airline_code}{flight_number}
  Route: {origin_name} → {dest_name}
  Date: {flight_date}
  Departure: {dep_time}
  Arrival: {arr_time}
  Duration: {elapsed_minutes // 60}h {elapsed_minutes % 60}m
  Distance: {distance} miles

PREDICTIONS:
{'='*50}
"""
        for model_name, pred_info in all_predictions.items():
            text_report += f"\n{model_name.replace('_', ' ')}:\n"
            text_report += f"  Result: {pred_info['result']}\n"
            text_report += f"  Confidence: {pred_info['confidence']:.1f}%\n"
        
        text_report += f"\n{'='*50}\n"
        text_report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        st.download_button(
            label="📄 Download as TXT",
            data=text_report,
            file_name=f"flight_prediction_{airline_code}{flight_number}_{flight_date}.txt",
            mime="text/plain"
        )

else:
    st.info("👈 Fill in the flight details and click 'Predict Delay' to get started!")
    
    # Show some statistics
    st.subheader("About This App")
    st.write("This app uses Machine Learning to predict flight delays based on:")
    st.write("- Airline and route information")
    st.write("- Departure time and date")
    st.write("- Flight distance and duration")
    st.write("- Historical delay patterns")
    
    # Show available airlines and airports
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Airlines", len(airlines_df))
    with col2:
        st.metric("Available Airports", len(airports_df))
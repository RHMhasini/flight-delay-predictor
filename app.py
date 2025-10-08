import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime, time
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# Title
st.title("✈️ Flight Delay Prediction System")
st.markdown("Predict flight delays using Machine Learning")
st.markdown("---")

# Load models and data
@st.cache_resource
def load_models():
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return xgb_model, label_encoders

@st.cache_data
def load_data():
    airlines = pd.read_csv('data/airlines.csv')
    airports = pd.read_csv('data/airports.csv')
    distance_lookup = pd.read_csv('data/distance_lookup.csv')
    
    with open('data/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    # Load model metrics and feature importance
    try:
        with open('data/model_metrics.json', 'r') as f:
            model_metrics = json.load(f)
    except:
        model_metrics = None
    
    try:
        with open('data/feature_importance.json', 'r') as f:
            feature_importance = json.load(f)
    except:
        feature_importance = None
    
    return airlines, airports, distance_lookup, feature_info, model_metrics, feature_importance

# Load everything
xgb_model, label_encoders = load_models()
airlines_df, airports_df, distance_lookup_df, feature_info, model_metrics, feature_importance = load_data()

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

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

def get_route_distance(origin, dest):
    """Get distance for a specific route from lookup table"""
    route = distance_lookup_df[
        (distance_lookup_df['ORIGIN'] == origin) & 
        (distance_lookup_df['DEST'] == dest)
    ]
    if not route.empty:
        return int(route.iloc[0]['DISTANCE'])
    return None

def create_gauge_chart(value, title):
    """Create a gauge chart for confidence score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#90EE90'},
                {'range': [50, 75], 'color': '#FFD700'},
                {'range': [75, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def create_feature_importance_chart(feature_data, top_n=10):
    """Create horizontal bar chart for feature importance"""
    df = pd.DataFrame({
        'Feature': feature_data['features'],
        'Importance': feature_data['importance']
    })
    df = df.sort_values('Importance', ascending=True).tail(top_n)
    
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def create_csv_export(flight_info, prediction):
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
        'Distance_Miles': [flight_info['distance']],
        'Prediction': [prediction['result']],
        'Confidence': [f"{prediction['confidence']:.1f}%"]
    }
    
    return pd.DataFrame(data)

def preprocess_input(airline_code, flight_num, origin, dest, dep_time, arr_time, 
                     elapsed_time, distance, flight_date):
    """Convert user inputs to model features"""
    
    day_of_week = flight_date.weekday()
    month = flight_date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    dep_hour = dep_time.hour
    time_block = get_time_block(dep_hour)
    
    crs_dep_time = dep_time.hour * 100 + dep_time.minute
    crs_arr_time = arr_time.hour * 100 + arr_time.minute
    
    distance_cat = get_distance_category(distance)
    
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
    
    for col in ['AIRLINE_CODE', 'ORIGIN', 'DEST', 'TIME_BLOCK', 'DISTANCE_CAT']:
        if col in label_encoders:
            df[col] = df[col].map(label_encoders[col])
    
    return df

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Prediction", "Prediction History"])

if page == "Prediction":
    # Sidebar - User Inputs
    st.sidebar.header("Flight Information")

    airline_name = st.sidebar.selectbox("Select Airline", airlines_df['AIRLINE_NAME'].values)
    airline_code = airlines_df[airlines_df['AIRLINE_NAME'] == airline_name]['AIRLINE_CODE'].values[0]

    flight_number = st.sidebar.number_input("Flight Number", min_value=1, max_value=9999, value=1234)
    flight_date = st.sidebar.date_input("Flight Date", datetime.now())

    airports_sorted = airports_df.sort_values('AIRPORT_NAME')
    origin_name = st.sidebar.selectbox("Origin Airport", airports_sorted['AIRPORT_NAME'].values, key='origin')
    origin_code = airports_df[airports_df['AIRPORT_NAME'] == origin_name]['AIRPORT_CODE'].values[0]

    dest_name = st.sidebar.selectbox("Destination Airport", airports_sorted['AIRPORT_NAME'].values, key='dest')
    dest_code = airports_df[airports_df['AIRPORT_NAME'] == dest_name]['AIRPORT_CODE'].values[0]

    # Auto-fill distance when airports are selected
    expected_distance = get_route_distance(origin_code, dest_code)
    
    if expected_distance is not None:
        default_distance = expected_distance
        distance_help = f"Auto-filled: {expected_distance} miles for this route"
    else:
        default_distance = 1000
        distance_help = "⚠️ Route not found in database. Please enter distance manually."
    
    distance = st.sidebar.number_input(
        "Distance (miles)", 
        min_value=0, 
        max_value=5000, 
        value=default_distance,
        help=distance_help
    )
    
    # Validate distance if we have expected value
    distance_warning = ""
    if expected_distance is not None and distance != expected_distance:
        tolerance = 0.20  # 20% tolerance
        min_distance = expected_distance * (1 - tolerance)
        max_distance = expected_distance * (1 + tolerance)
        
        if distance < min_distance or distance > max_distance:
            distance_warning = f"⚠️ Distance seems unusual. Expected ~{expected_distance} miles for this route."
            st.sidebar.warning(distance_warning)
        elif distance != expected_distance:
            st.sidebar.info(f"ℹ️ Modified from expected {expected_distance} miles")

    dep_time = st.sidebar.time_input("Departure Time", time(14, 30))
    arr_time = st.sidebar.time_input("Arrival Time", time(17, 0))

    elapsed_minutes = (arr_time.hour * 60 + arr_time.minute) - (dep_time.hour * 60 + dep_time.minute)
    if elapsed_minutes < 0:
        elapsed_minutes += 24 * 60

    is_valid_flight = True
    validation_message = ""

    if elapsed_minutes < 30:
        is_valid_flight = False
        validation_message = "Flight duration is less than 30 minutes. Please check your departure and arrival times."
    elif elapsed_minutes > 18 * 60:
        is_valid_flight = False
        validation_message = "Flight duration exceeds 18 hours. This seems unrealistic for most flights. Please verify your times."

    if not is_valid_flight:
        st.sidebar.error(f"❌ {validation_message}")
    else:
        st.sidebar.success(f"✓ Flight duration: {elapsed_minutes // 60}h {elapsed_minutes % 60}m")

    if origin_code == dest_code:
        is_valid_flight = False
        st.sidebar.error("❌ Origin and destination airports cannot be the same.")

    predict_button = st.sidebar.button("🔮 Predict Delay", type="primary", disabled=not is_valid_flight)

    # Main area
    if predict_button:
        input_df = preprocess_input(
            airline_code, flight_number, origin_code, dest_code,
            dep_time, arr_time, elapsed_minutes, distance, flight_date
        )
        
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
        
        # XGBoost Prediction
        st.subheader("🤖 XGBoost Prediction")
        xgb_pred = xgb_model.predict(input_df)[0]
        xgb_proba = xgb_model.predict_proba(input_df)[0]
        xgb_result = "DELAYED" if xgb_pred == 1 else "ON-TIME"
        xgb_confidence = xgb_proba[1] if xgb_pred == 1 else xgb_proba[0]
        
        prediction_data = {
            'result': xgb_result,
            'confidence': xgb_confidence * 100
        }
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if xgb_result == "DELAYED":
                st.error(f"🔴 **{xgb_result}**")
            else:
                st.success(f"🟢 **{xgb_result}**")
        with col2:
            st.plotly_chart(create_gauge_chart(xgb_confidence * 100, "Confidence Score"), use_container_width=True)
        
        # Feature importance
        if feature_importance:
            st.markdown("---")
            st.subheader("🔍 Feature Importance Analysis")
            st.plotly_chart(create_feature_importance_chart(feature_importance), use_container_width=True)
        
        # Add to prediction history
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'flight': f"{airline_code}{flight_number}",
            'route': f"{origin_name} → {dest_name}",
            'result': xgb_result,
            'confidence': xgb_confidence * 100
        }
        st.session_state.prediction_history.insert(0, history_entry)
        if len(st.session_state.prediction_history) > 10:
            st.session_state.prediction_history.pop()
        
        # Export section
        st.markdown("---")
        st.subheader("📥 Export Results")
        
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
            csv_data = create_csv_export(flight_info, prediction_data)
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

PREDICTION:
{'='*50}
  Result: {xgb_result}
  Confidence: {xgb_confidence * 100:.1f}%

{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            st.download_button(
                label="📄 Download as TXT",
                data=text_report,
                file_name=f"flight_prediction_{airline_code}{flight_number}_{flight_date}.txt",
                mime="text/plain"
            )

    else:
        st.info("👈 Fill in the flight details and click 'Predict Delay' to get started!")
        
        st.subheader("About This App")
        st.write("This app uses XGBoost Machine Learning to predict flight delays based on:")
        st.write("- ✈️ Airline and route information")
        st.write("- 🕒 Departure time and date")
        st.write("- 📏 Flight distance and duration")
        st.write("- 📊 Historical delay patterns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Available Airlines", len(airlines_df))
        with col2:
            st.metric("Available Airports", len(airports_df))
        with col3:
            st.metric("Known Routes", len(distance_lookup_df))

elif page == "Prediction History":
    st.header("📝 Prediction History")
    
    if st.session_state.prediction_history:
        for i, entry in enumerate(st.session_state.prediction_history):
            with st.expander(f"{entry['timestamp']} - {entry['flight']} ({entry['route']})"):
                col1, col2 = st.columns(2)
                with col1:
                    if entry['result'] == "DELAYED":
                        st.error(f"🔴 **{entry['result']}**")
                    else:
                        st.success(f"🟢 **{entry['result']}**")
                with col2:
                    st.metric("Confidence", f"{entry['confidence']:.1f}%")
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet. Make a prediction to see history here.")
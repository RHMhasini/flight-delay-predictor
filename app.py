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
    
    return airlines, airports, feature_info, model_metrics, feature_importance

# Load everything
xgb_model, log_data, label_encoders = load_models()
airlines_df, airports_df, feature_info, model_metrics, feature_importance = load_data()

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

def create_confidence_comparison(predictions):
    """Create bar chart comparing model confidences"""
    models = list(predictions.keys())
    confidences = [predictions[m]['confidence'] for m in models]
    
    colors = ['#3498db' if predictions[m]['result'] == 'ON-TIME' else '#e74c3c' for m in models]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=confidences,
            marker_color=colors,
            text=[f"{c:.1f}%" for c in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Model Confidence Comparison",
        xaxis_title="Model",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        showlegend=False
    )
    
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

def create_metrics_comparison(metrics_data):
    """Create grouped bar chart for model metrics"""
    models = list(metrics_data.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [metrics_data[model][metric] * 100 for model in models]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=models,
            y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Model",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

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
    
    for model_name, pred_info in predictions.items():
        data[f'{model_name}_Prediction'] = [pred_info['result']]
        data[f'{model_name}_Confidence'] = [f"{pred_info['confidence']:.1f}%"]
    
    return pd.DataFrame(data)

def preprocess_input(airline_code, flight_num, origin, dest, dep_time, arr_time, 
                     elapsed_time, distance, flight_date):
    """Convert user inputs to model features"""
    
    day_of_week = flight_date.weekday()
    month = flight_date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    dep_hour = dep_time.hour
    time_block = get_time_block(dep_hour)
    crs_dep_time = dep_hour + (dep_time.minute / 60)
    crs_arr_time = arr_time.hour + (arr_time.minute / 60)
    
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
page = st.sidebar.radio("Navigation", ["Prediction", "Model Performance", "Prediction History"])

if page == "Prediction":
    # Sidebar - User Inputs
    st.sidebar.header("Flight Information")

    airline_name = st.sidebar.selectbox("Select Airline", airlines_df['AIRLINE_NAME'].values)
    airline_code = airlines_df[airlines_df['AIRLINE_NAME'] == airline_name]['AIRLINE_CODE'].values[0]

    flight_number = st.sidebar.number_input("Flight Number", min_value=1, max_value=9999, value=1234)
    flight_date = st.sidebar.date_input("Flight Date", datetime.now())

    airports_sorted = airports_df.sort_values('AIRPORT_NAME')
    origin_name = st.sidebar.selectbox("Origin Airport", airports_sorted['AIRPORT_NAME'].values)
    origin_code = airports_df[airports_df['AIRPORT_NAME'] == origin_name]['AIRPORT_CODE'].values[0]

    dest_name = st.sidebar.selectbox("Destination Airport", airports_sorted['AIRPORT_NAME'].values)
    dest_code = airports_df[airports_df['AIRPORT_NAME'] == dest_name]['AIRPORT_CODE'].values[0]

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

    distance = st.sidebar.number_input("Distance (miles)", min_value=0, max_value=5000, value=1000)

    if origin_code == dest_code:
        is_valid_flight = False
        st.sidebar.error("❌ Origin and destination airports cannot be the same.")

    model_choice = st.sidebar.selectbox("Select Model", ["XGBoost", "Logistic Regression", "Both Models"])

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
        
        all_predictions = {}
        
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
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if xgb_result == "DELAYED":
                    st.error(f"🔴 {xgb_result}")
                else:
                    st.success(f"🟢 {xgb_result}")
            with col2:
                st.plotly_chart(create_gauge_chart(xgb_confidence * 100, "Confidence"), use_container_width=True)
            
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
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if log_result == "DELAYED":
                    st.error(f"🔴 {log_result}")
                else:
                    st.success(f"🟢 {log_result}")
            with col2:
                st.plotly_chart(create_gauge_chart(log_confidence * 100, "Confidence"), use_container_width=True)
        
        # Confidence comparison if both models
        if len(all_predictions) > 1:
            st.markdown("---")
            st.subheader("📊 Model Comparison")
            st.plotly_chart(create_confidence_comparison(all_predictions), use_container_width=True)
        
        # Feature importance
        if feature_importance:
            st.markdown("---")
            st.subheader("🔍 Feature Importance")
            st.plotly_chart(create_feature_importance_chart(feature_importance), use_container_width=True)
        
        # Add to prediction history
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'flight': f"{airline_code}{flight_number}",
            'route': f"{origin_name} → {dest_name}",
            'predictions': all_predictions
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
        
        st.subheader("About This App")
        st.write("This app uses Machine Learning to predict flight delays based on:")
        st.write("- Airline and route information")
        st.write("- Departure time and date")
        st.write("- Flight distance and duration")
        st.write("- Historical delay patterns")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Available Airlines", len(airlines_df))
        with col2:
            st.metric("Available Airports", len(airports_df))

elif page == "Model Performance":
    st.header("📈 Model Performance Metrics")
    
    if model_metrics:
        st.plotly_chart(create_metrics_comparison(model_metrics), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### XGBoost")
            for metric, value in model_metrics['XGBoost'].items():
                st.metric(metric.replace('_', ' ').title(), f"{value*100:.2f}%")
        
        with col2:
            st.markdown("### Logistic Regression")
            for metric, value in model_metrics['Logistic_Regression'].items():
                st.metric(metric.replace('_', ' ').title(), f"{value*100:.2f}%")
    else:
        st.warning("Model metrics not available. Please export model_metrics.json from your training notebook.")

elif page == "Prediction History":
    st.header("📝 Prediction History")
    
    if st.session_state.prediction_history:
        for i, entry in enumerate(st.session_state.prediction_history):
            with st.expander(f"{entry['timestamp']} - {entry['flight']} ({entry['route']})"):
                for model, pred in entry['predictions'].items():
                    st.write(f"**{model}:** {pred['result']} (Confidence: {pred['confidence']:.1f}%)")
        
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet. Make a prediction to see history here.")
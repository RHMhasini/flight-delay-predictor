# Flight Delay Predictor

A machine learning-based web application that predicts whether a flight will be delayed (arrival delay > 15 minutes) using historical flight data.

## Project Overview

Flight Delay Predictor uses an XGBoost model to forecast flight delays based on airline, route, time, and date. The application offers real-time predictions, visualizations of feature importance, and tracks prediction history.

### Key Features
- Real-time predictions via an interactive Streamlit web interface
- XGBoost model achieving 74.8% accuracy
- Feature importance visualization to understand factors affecting delays
- Prediction history tracking
- Export options (CSV and TXT formats)
- Input validation and auto-fill features

## Project Structure
flight-delay-predictor/
├── app.py # Main Streamlit application
├── test_model.py # Model testing script
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
├── colab_code/
│ └── colab_code.py # Complete ML pipeline (data processing, training, evaluation)
├── models/
│ ├── xgboost_model.pkl # Trained XGBoost model
│ ├── logistic_model.pkl # Trained Logistic Regression model (with scaler)
│ └── label_encoders.pkl # Label encoders for categorical features
└── data/
├── airlines.csv # Airline code to name mapping
├── airports.csv # Airport code to name mapping
├── distance_lookup.csv # Route distance lookup table
├── feature_info.json # Feature statistics and metadata
├── feature_importance.json # Feature importance scores
├── feature_names.txt # List of all features
└── model_metrics.json # Model performance metrics


## Data Processing

- **Source:** US Department of Transportation flight data from kaggle (~3M records)
- **Target:** Binary classification (ON-TIME vs DELAYED, threshold = 15 min)
- **Preprocessing:** Removed leakage features and cleaned missing data
- **Feature Engineering:** Temporal features (day, month, weekend, time block), route features (distance categories)
- **Encoding & Scaling:** Label encoding for categorical features, standard scaling for numerical features used in Logistic Regression

## Machine Learning Models

### XGBoost Classifier (Primary)
- Accuracy: 74.8%
- Feature Importance (Top 5): Departure Hour, Scheduled Departure Time, Month, Airline, Origin Airport

### Logistic Regression (Baseline)
- Accuracy: 54.6%

### Random Forest (Secondary)
- Trained but excluded from repo due to file size (>100MB)

## Web Application (Streamlit)

- Collects flight information from users
- Validates input and calculates derived features
- Predicts delays using the XGBoost model
- Displays results with confidence and feature importance charts
- Tracks the last 10 predictions

### Key Functions
- `preprocess_input()`: Converts user input into model-ready features
- `get_route_distance()`: Looks up route distances
- `create_gauge_chart()`: Visualizes prediction confidence
- `create_feature_importance_chart()`: Shows top 10 important features

## How to Use
 1. Clone the repository
  git clone https://github.com/RHMhasini/flight-delay-predictor.git
  cd flight-delay-predictor

 2. Install dependencies
  pip install -r requirements.txt

 3. Run the application
  streamlit run app.py

 4. Access the app
  Open your browser and go to http://localhost:8501

Dependencies:
- streamlit   - scikit-learn
- pandas      - xgboost
- numpy       - plotly

Summary:
Features: 14  |  Models Trained: 3 (XGBoost, Random Forest, Logistic Regression)
Training Data: ~2.4M flights  |  Airlines: 18  |  Airports: 381

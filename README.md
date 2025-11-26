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
| **Directory / File**           | **Type**  | **Description**                                         |
| ------------------------------ | --------- | ------------------------------------------------------- |
| `app.py`                       | File      | Main Streamlit web application                          |
| `test_model.py`                | File      | Script to test models and data loading                  |
| `requirements.txt`             | File      | Python dependencies                                     |
| `.gitignore`                   | File      | Git ignore rules                                        |
| `colab_code/`                  | Directory | Contains full ML pipeline for training and evaluation   |
| `colab_code/colab_code.py`     | File      | Data preprocessing, feature engineering, model training |
| `models/`                      | Directory | Stores trained models and encoders                      |
| `models/xgboost_model.pkl`     | File      | Trained XGBoost classifier                              |
| `models/logistic_model.pkl`    | File      | Trained Logistic Regression model (with scaler)         |
| `models/label_encoders.pkl`    | File      | Saved label encoders for categorical features           |
| `data/`                        | Directory | Contains supporting data files                          |
| `data/airlines.csv`            | File      | Airline code to name mapping                            |
| `data/airports.csv`            | File      | Airport code to name mapping                            |
| `data/distance_lookup.csv`     | File      | Route distance lookup table                             |
| `data/feature_info.json`       | File      | Feature statistics and metadata                         |
| `data/feature_importance.json` | File      | Feature importance scores                               |
| `data/feature_names.txt`       | File      | List of all features                                    |
| `data/model_metrics.json`      | File      | Model performance metrics                               |



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

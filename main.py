import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import importlib.util

# Add custom CSS for the weather dashboard theme
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0b0f14, #1a1f26);
        color: #fff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f14, #1a1f26);
    }
    
    .weather-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-item {
        display: flex;
        justify-content: space-between;
        margin: 10px 0;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .temp-display {
        font-size: 48px;
        font-weight: 300;
        color: #ffb13b;
        text-align: center;
        margin: 20px 0;
    }
    
    .condition-text {
        font-size: 24px;
        opacity: 0.8;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #ffb13b, #ff8c00);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: 500;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(255, 177, 59, 0.3);
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 177, 59, 0.4);
    }
    
    h1, h2, h3 {
        color: #fff;
    }
    
    .css-1d391kg, .css-12oz5g7 {
        background: linear-gradient(135deg, #0b0f14, #1a1f26);
    }
    
    /* Custom prediction display */
    .prediction-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .prediction-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-row:last-child {
        border-bottom: none;
    }
    
    .feature-input {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Dynamic import function for your modules
def import_module_from_file(file_path, module_name):
    """Import a module from a file path"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        st.error(f"Error importing {module_name}: {e}")
        return None

# Import your modules
evaluate_module = import_module_from_file("src/evaluate/evaluate.py", "evaluate")
predict_module = import_module_from_file("src/predict/predict.py", "predict")
test_predict_module = import_module_from_file("src/predict/test_predict.py", "test_predict")

# Cache the model loading
@st.cache_resource
def load_models():
    """Load models using the function from evaluate.py"""
    try:
        if evaluate_module and hasattr(evaluate_module, 'load_models_and_data'):
            models, X_test, y_test, X_train, y_train = evaluate_module.load_models_and_data()
            return models, X_test, y_test, X_train, y_train
        else:
            st.error("Evaluate module not loaded properly")
            return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

# Cache the TemperaturePredictor
@st.cache_resource
def load_temperature_predictor():
    """Load the TemperaturePredictor from test_predict.py"""
    try:
        if test_predict_module and hasattr(test_predict_module, 'TemperaturePredictor'):
            predictor = test_predict_module.TemperaturePredictor()
            return predictor
        else:
            st.error("TemperaturePredictor not found in test_predict.py")
            return None
    except Exception as e:
        st.error(f"Error loading TemperaturePredictor: {e}")
        return None

# Enhanced prediction function that uses methods from predict.py
def get_predictions_with_details():
    """Get predictions with detailed weather information"""
    try:
        if not predict_module:
            st.error("Predict module not loaded")
            return None, None
            
        # Get real-time weather data
        weather_data = predict_module.get_realtime_weather()
        
        if not weather_data:
            return None, None
        
        # Get models for prediction
        models_obj, _, _, _, _ = load_models()
        if not models_obj:
            return None, None
        
        # Convert to features (using function from predict.py)
        X = predict_module.convert_to_features(weather_data)
        
        # Get predictions from all models
        predictions = {}
        model_names = {
            'ols': 'OLS Regression',
            'bgd': 'Batch Gradient Descent', 
            'sgd': 'Stochastic Gradient Descent'
        }
        
        for key, model in models_obj.items():
            pred = model.predict(X)[0]
            predictions[model_names[key]] = pred
        
        return weather_data, predictions
        
    except Exception as e:
        st.error(f"Error getting predictions: {e}")
        return None, None

# Function to run model evaluation
def run_model_evaluation():
    """Run the complete model evaluation from evaluate.py"""
    try:
        if evaluate_module and hasattr(evaluate_module, 'evaluate_all_models'):
            evaluation_results = evaluate_module.evaluate_all_models()
            return evaluation_results
        else:
            st.error("Evaluate module not loaded properly")
            return None
    except Exception as e:
        st.error(f"Error during evaluation: {e}")
        return None

# Function to get manual predictions
def get_manual_predictions(features):
    """Get predictions using manually entered features"""
    try:
        predictor = load_temperature_predictor()
        if predictor:
            predictions = predictor.predict_temperature(features)
            
            # Convert model names to readable format
            readable_predictions = {}
            model_names = {
                'ols': 'OLS Regression',
                'bgd': 'Batch Gradient Descent', 
                'sgd': 'Stochastic Gradient Descent'
            }
            
            for key, value in predictions.items():
                readable_name = model_names.get(key, key.upper())
                readable_predictions[readable_name] = value
            
            return readable_predictions
        else:
            st.error("Temperature predictor not loaded")
            return None
    except Exception as e:
        st.error(f"Error getting manual predictions: {e}")
        return None

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Weather Dashboard",
        page_icon="🌤️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    inject_custom_css()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #ffb13b;'>Weather Dashboard</h1>", 
                   unsafe_allow_html=True)
        # Developers
        st.markdown(
            """
            <p style='text-align: center; font-size: 14px; color: #999; margin-top: -10px;'>
            Developed by: <b>biniyam_girma</b> (ATE/7146/14),
            <b>simon_mesfin</b> (ATE/7211/14),
            <b>yosef_ashebir</b> (ATE/4638/14)
            </p>
            """,
            unsafe_allow_html=True
        )
        current_time = datetime.now().strftime("%A, %B %d, %Y %H:%M")
        st.markdown(f"<p style='text-align: center; opacity: 0.8;'>{current_time}</p>", 
                   unsafe_allow_html=True)
    
    # Check if modules loaded successfully
    if not evaluate_module or not predict_module or not test_predict_module:
        st.error("Failed to load required modules. Please check your file structure.")
        st.info("""
        Expected folder structure:
        ```
        your_project/
        ├── app.py
        ├── src/
        │   ├── evaluate/
        │   │   └── evaluate.py
        │   └── predict/
        │       ├── predict.py
        │       └── test_predict.py
        ├── models/
        │   └── regression/
        │       ├── ols_model.pkl
        │       ├── bgd_model.pkl
        │       └── sgd_model.pkl
        └── data/
            └── processed/
                └── regression_data.csv
        ```
        """)
        return
    
    # Load models at startup
    models_obj, X_test, y_test, X_train, y_train = load_models()
    
    if models_obj is None:
        st.error("Failed to load models. Please ensure models are trained first.")
        st.info("Run train.py first to train the models, then restart this app.")
        return
    
    # Main content - Updated tabs to include Manual Prediction
    tab1, tab2, tab3, tab4 = st.tabs(["🌤️ Live Weather", "🔢 Manual Prediction", "📊 Model Evaluation", "🔍 Model Comparison"])
    
    with tab1:
        st.markdown("<div class='weather-card'>", unsafe_allow_html=True)
        st.markdown("<h2>Real-time Weather Prediction</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔄 Fetch Current Weather & Predict", key="predict_btn"):
                with st.spinner("Fetching weather data and generating predictions..."):
                    weather_data, predictions = get_predictions_with_details()
                    
                    if weather_data and predictions:
                        # Display current weather
                        st.markdown(f"<div class='temp-display'>{weather_data['actual_temp']:.4f}°C</div>", 
                                   unsafe_allow_html=True)
                        
                        condition = weather_data.get('condition', 'Unknown')
                        location = weather_data.get('location', 'Addis Ababa')
                        country = weather_data.get('country', 'ET')
                        
                        st.markdown(f"<div class='condition-text'>{condition} in {location}, {country}</div>", 
                                   unsafe_allow_html=True)
                        
                        # Weather details
                        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                        
                        with detail_col1:
                            st.metric("Humidity", f"{weather_data['humidity']*100:.1f}%")
                        with detail_col2:
                            st.metric("Wind Speed", f"{weather_data['wind_speed']:.1f} km/h")
                        with detail_col3:
                            st.metric("Pressure", f"{weather_data['pressure']:.0f} hPa")
                        with detail_col4:
                            st.metric("Visibility", f"{weather_data['visibility']:.1f} km")
                        
                        # Predictions in a styled container
                        st.markdown("<h3>Model Predictions</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
                        
                        for model_name, prediction in predictions.items():
                            st.markdown(
                                f"""
                                <div class='prediction-row'>
                                    <span>{model_name}</span>
                                    <span style='color: #ffb13b; font-weight: 500;'>{prediction:.4f}°C</span>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Actual vs predicted comparison
                        actual_temp = weather_data['actual_temp']
                        best_pred = min(predictions.values(), key=lambda x: abs(x - actual_temp))
                        best_model = [k for k, v in predictions.items() if v == best_pred][0]
                        accuracy = abs(actual_temp - best_pred)
                        
                        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🌡️ Actual Temperature", f"{actual_temp:.4f}°C")
                        with col2:
                            st.metric("🎯 Most Accurate", 
                                     f"{best_pred:.4f}°C", 
                                     f"{accuracy:.4f}°C diff - {best_model}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("Failed to get weather data or predictions. Please try again.")
        
        with col2:
            st.markdown("<h3>Location Info</h3>", unsafe_allow_html=True)
            st.write("📍 Default: Addis Ababa, Ethiopia")
            st.write("🌍 Latitude: 9.03°")
            st.write("🌍 Longitude: 38.74°")
            
            st.markdown("<h3 style='margin-top: 20px;'>About Predictions</h3>", unsafe_allow_html=True)
            st.write("""
            The models predict temperature based on:
            - Current weather conditions
            - Time of day and month
            - Historical patterns
            - Machine learning algorithms
            """)
            
            st.markdown("<h3>Models Used</h3>", unsafe_allow_html=True)
            st.write("• OLS Regression")
            st.write("• Batch Gradient Descent")
            st.write("• Stochastic Gradient Descent")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='weather-card'>", unsafe_allow_html=True)
        st.markdown("<h2>Manual Weather Prediction</h2>", unsafe_allow_html=True)
        st.markdown("<p>Enter weather features manually to get temperature predictions from all models.</p>", 
                   unsafe_allow_html=True)
        
        # Feature input form
        with st.form("manual_prediction_form"):
            st.markdown("<div class='feature-input'>", unsafe_allow_html=True)
            st.markdown("<h3>Weather Features</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                apparent_temp = st.number_input(
                    "Apparent Temperature (°C)",
                    min_value=-50.0,
                    max_value=50.0,
                    value=7.28,
                    step=0.1,
                    help="How temperature actually feels"
                )
                
                humidity = st.slider(
                    "Humidity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.86,
                    step=0.01,
                    help="Relative humidity (0.0 to 1.0)"
                )
                
                wind_speed = st.number_input(
                    "Wind Speed (km/h)",
                    min_value=0.0,
                    max_value=200.0,
                    value=14.26,
                    step=0.1,
                    help="Wind speed in kilometers per hour"
                )
                
                wind_bearing = st.number_input(
                    "Wind Bearing (degrees)",
                    min_value=0.0,
                    max_value=360.0,
                    value=259.0,
                    step=1.0,
                    help="Wind direction in degrees (0-360)"
                )
            
            with col2:
                visibility = st.number_input(
                    "Visibility (km)",
                    min_value=0.0,
                    max_value=50.0,
                    value=15.83,
                    step=0.1,
                    help="Visibility distance in kilometers"
                )
                
                pressure = st.number_input(
                    "Pressure (millibars)",
                    min_value=800.0,
                    max_value=1100.0,
                    value=1015.63,
                    step=0.1,
                    help="Atmospheric pressure in millibars"
                )
                
                month = st.slider(
                    "Month",
                    min_value=1,
                    max_value=12,
                    value=3,
                    step=1,
                    help="Month of the year (1-12)"
                )
                
                hour = st.slider(
                    "Hour",
                    min_value=0,
                    max_value=23,
                    value=23,
                    step=1,
                    help="Hour of the day (0-23)"
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("🌡️ Predict Temperature", use_container_width=True)
            
            if submitted:
                # Prepare features dictionary
                features = {
                    'Apparent Temperature (C)': apparent_temp,
                    'Humidity': humidity,
                    'Wind Speed (km/h)': wind_speed,
                    'Wind Bearing (degrees)': wind_bearing,
                    'Visibility (km)': visibility,
                    'Pressure (millibars)': pressure,
                    'month': month,
                    'hour': hour
                }
                
                with st.spinner("Generating predictions from all models..."):
                    predictions = get_manual_predictions(features)
                    
                    if predictions:
                        # Display predictions
                        st.markdown("<h3>Model Predictions</h3>", unsafe_allow_html=True)
                        
                        # Calculate average prediction
                        avg_prediction = sum(predictions.values()) / len(predictions)
                        
                        st.markdown(f"<div class='temp-display'>{avg_prediction:.4f}°C</div>", 
                                   unsafe_allow_html=True)
                        st.markdown(f"<div class='condition-text'>Average Predicted Temperature</div>", 
                                   unsafe_allow_html=True)
                        
                        # Display individual model predictions
                        st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
                        
                        for model_name, prediction in predictions.items():
                            st.markdown(
                                f"""
                                <div class='prediction-row'>
                                    <span>{model_name}</span>
                                    <span style='color: #ffb13b; font-weight: 500;'>{prediction:.4f}°C</span>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Model comparison
                        st.markdown("<h3>Model Comparison</h3>", unsafe_allow_html=True)
                        best_model = min(predictions.items(), key=lambda x: abs(x[1] - avg_prediction))
                        worst_model = max(predictions.items(), key=lambda x: abs(x[1] - avg_prediction))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "🤝 Most Consistent", 
                                f"{best_model[1]:.4f}°C", 
                                best_model[0]
                            )
                        with col2:
                            st.metric(
                                "⚡ Most Divergent", 
                                f"{worst_model[1]:.4f}°C", 
                                worst_model[0]
                            )
                        
                        # Show features used
                        with st.expander("View Features Used for Prediction"):
                            feature_df = pd.DataFrame([features])
                            st.dataframe(feature_df)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='weather-card'>", unsafe_allow_html=True)
        st.markdown("<h2>Model Performance Evaluation</h2>", unsafe_allow_html=True)
        
        if st.button("Run Comprehensive Evaluation", key="eval_btn"):
            with st.spinner("Running model evaluation... This may take a few moments."):
                evaluation_results = run_model_evaluation()
                
                if evaluation_results is not None:
                    results, comparison_df = evaluation_results
                    
                    # Display results
                    st.markdown("<h3>Evaluation Metrics</h3>", unsafe_allow_html=True)
                    
                    # Display the comparison dataframe
                    st.dataframe(comparison_df.style.format({
                        'MSE': '{:.4f}',
                        'RMSE': '{:.4f}', 
                        'MAE': '{:.4f}',
                        'R²': '{:.4f}'
                    }))
                    
                    # Best models
                    best_mse = min(results, key=lambda x: x['mse'])
                    best_r2 = max(results, key=lambda x: x['r2'])
                    best_mae = min(results, key=lambda x: x['mae'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏆 Best (MSE)", best_mse['model_name'], f"{best_mse['mse']:.4f}")
                    with col2:
                        st.metric("🏆 Best (R²)", best_r2['model_name'], f"{best_r2['r2']:.4f}")
                    with col3:
                        st.metric("🏆 Best (MAE)", best_mae['model_name'], f"{best_mae['mae']:.4f}")
                    
                    # Visualizations
                    st.markdown("<h3>Model Performance Visualizations</h3>", unsafe_allow_html=True)
                    try:
                        # List of visualization files to display
                        viz_files = [
                            "visuals/regression/all_models_comparison.png",
                            "visuals/regression/ols_detailed_analysis.png", 
                            "visuals/regression/bgd_detailed_analysis.png",
                            "visuals/regression/sgd_detailed_analysis.png"
                        ]
                        
                        # Check which files exist and display them
                        for viz_file in viz_files:
                            if os.path.exists(viz_file):
                                st.image(viz_file, caption=os.path.basename(viz_file).replace('.png', '').replace('_', ' ').title())
                            else:
                                st.warning(f"Visualization not found: {viz_file}")
                                
                    except Exception as e:
                        st.error(f"Error loading visualizations: {e}")
                        st.info("Run evaluate.py directly to generate visualization files first")
                
                else:
                    st.error("Model evaluation failed. Please check the console for errors.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<div class='weather-card'>", unsafe_allow_html=True)
        st.markdown("<h2>Model Details & Comparison</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        ### About the Models
        
        **OLS Regression (Ordinary Least Squares)**
        - Traditional linear regression approach
        - Minimizes sum of squared residuals
        - Best for smaller datasets
        - Provides exact mathematical solution
        
        **Batch Gradient Descent**
        - Uses entire dataset for each parameter update
        - Stable and smooth convergence
        - Computationally expensive for large datasets
        - Guaranteed convergence to local minimum
        
        **Stochastic Gradient Descent**
        - Uses single random sample for each update
        - Faster convergence, better for large datasets
        - Noisy but efficient updates
        - Can escape local minima
        """)
        
        # Test set information
        st.markdown("<h3>Evaluation Dataset</h3>", unsafe_allow_html=True)
        if X_test is not None and y_test is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Samples", X_test.shape[0])
            with col2:
                st.metric("Features", X_test.shape[1])
            with col3:
                st.metric("Data Split", "80% Train / 20% Test")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
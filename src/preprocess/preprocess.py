import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def preprocess_data():
    """
    Preprocess weather data for linear regression task
    Predicts Temperature (C) based on weather features
    """
    try:
        # Load raw data
        print("Loading raw data...")
        df = pd.read_csv("data/raw/weatherHistory.csv")
        
        # Handle missing values
        print("Handling missing values...")
        df = df.dropna(subset=['Precip Type'])  # Remove rows with missing precipitation type
        
        # Feature engineering
        print("Performing feature engineering...")
        df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
        df['month'] = df['Formatted Date'].dt.month
        df['hour'] = df['Formatted Date'].dt.hour
        df['day_of_year'] = df['Formatted Date'].dt.dayofyear
        
        # Select features for linear regression
        # Predicting Temperature (C) based on other weather features
        regression_features = [
            'Apparent Temperature (C)', 
            'Humidity', 
            'Wind Speed (km/h)', 
            'Wind Bearing (degrees)', 
            'Visibility (km)', 
            'Pressure (millibars)',
            'month', 
            'hour'
        ]
        
        # Create target variable and features
        target = 'Temperature (C)'
        
        # Ensure all features exist in dataframe
        available_features = [f for f in regression_features if f in df.columns]
        print(f"Using features: {available_features}")
        
        regression_df = df[available_features + [target]].copy()
        
        # Remove any remaining missing values
        regression_df = regression_df.dropna()
        
        # Save processed data
        print("Saving processed data...")
        os.makedirs("data/processed", exist_ok=True)
        regression_df.to_csv("data/processed/regression_data.csv", index=False)
        
        # Print dataset info
        print(f"Processed dataset shape: {regression_df.shape}")
        print(f"Features: {available_features}")
        print(f"Target: {target}")
        print(f"Data saved to: data/processed/regression_data.csv")
        
        return regression_df
        
    except FileNotFoundError:
        print("Error: Raw data file not found at 'data/raw/weatherHistory.csv'")
        print("Please ensure the raw data file exists in the correct location")
        return None
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting data preprocessing for linear regression...")
    processed_data = preprocess_data()
    
    if processed_data is not None:
        print("Data preprocessing completed successfully!")
    else:
        print("Data preprocessing failed!")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def train_ols_regression(X_train, y_train):
    """Train Linear Regression using Ordinary Least Squares (Normal Equation)"""
    print("Training OLS Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_bgd_regression(X_train, y_train, max_iter=1000, tol=1e-6, learning_rate=0.001):
    """Train Linear Regression using Batch Gradient Descent (scaled + SGDRegressor)"""
    print("Training Batch Gradient Descent (Pipeline + Scaling)...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDRegressor(
            loss="squared_error",   # Linear Regression Loss
            penalty=None,           # No L2 regularization
            learning_rate="constant",
            eta0=learning_rate,     # Learning rate
            max_iter=max_iter,      # Epochs
            tol=tol,
            shuffle=False           # TRUE Batch Gradient Descent (no shuffle)
        ))
    ])

    model.fit(X_train, y_train)
    return model

def train_sgd_regression(X_train, y_train, max_iter=1000):
    """Train Linear Regression using Stochastic Gradient Descent (Pipeline + Scaling)"""
    print("Training Stochastic Gradient Descent...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDRegressor(
            loss="squared_error",
            penalty=None,                # pure SGD (no L2)
            learning_rate="adaptive",    # adjusts learning rate automatically
            eta0=0.01,                   # initial LR
            max_iter=max_iter,           # epochs
            tol=None,                    # do not stop early
            shuffle=True,                # REQUIRED for true SGD
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    return model

def save_models(ols_model, bgd_model, sgd_model):
    """Save all trained models"""
    print("\nSaving models...")
    
    # Ensure models directory exists
    os.makedirs("models/regression", exist_ok=True)
    
    # Save models
    joblib.dump(ols_model, "models/regression/ols_model.pkl")
    joblib.dump(bgd_model, "models/regression/bgd_model.pkl")
    joblib.dump(sgd_model, "models/regression/sgd_model.pkl")
    
    print("Models saved successfully!")
    print("OLS model: models/regression/ols_model.pkl")
    print("BGD model: models/regression/bgd_model.pkl")
    print("SGD model: models/regression/sgd_model.pkl")

def load_training_data():
    """Load and prepare training data"""
    print("Loading processed regression data...")
    
    # Load processed regression dataset
    reg_df = pd.read_csv("data/processed/regression_data.csv")
    print(f"Dataset shape: {reg_df.shape}")
    
    # Features and target
    X = reg_df.drop('Temperature (C)', axis=1)
    y = reg_df['Temperature (C)']
    
    print(f"Features: {list(X.columns)}")
    print(f"Target: Temperature (C)")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_linear_regression_models():
    """Train three different linear regression approaches and save models"""
    try:
        # Load training data
        X_train, X_test, y_train, y_test = load_training_data()
        
        # Train OLS Regression (Normal Equation)
        print("\n" + "="*50)
        ols_model = train_ols_regression(X_train, y_train)
        
        # Train Batch Gradient Descent
        print("\n" + "="*50)
        bgd_model = train_bgd_regression(X_train, y_train)
        
        # Train Stochastic Gradient Descent
        print("\n" + "="*50)
        sgd_model = train_sgd_regression(X_train, y_train)
        
        # Save all models
        save_models(ols_model, bgd_model, sgd_model)
        
        # Return models and test data for potential immediate evaluation
        return {
            'models': {
                'ols': ols_model,
                'bgd': bgd_model,
                'sgd': sgd_model
            },
            'test_data': {
                'X_test': X_test,
                'y_test': y_test
            }
        }
        
    except FileNotFoundError:
        print("Error: Processed data not found. Please run preprocessing first.")
        return None
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting Linear Regression Models Training...")
    print("="*60)
    
    result = train_linear_regression_models()
    
    if result is not None:
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("All models have been trained and saved.")
        print("Run evaluate.py to see model performance metrics.")
    else:
        print("\nTraining failed!")
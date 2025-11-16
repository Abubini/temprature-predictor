# 1============================================================================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_linear_regression():
    print("Training Linear Regression model...")

    # Load processed regression dataset
    reg_df = pd.read_csv("data/processed/regression_data.csv")

    # Features and target
    X_reg = reg_df.drop('Temperature (C)', axis=1)
    y_reg = reg_df['Temperature (C)']

    # Split into train and test sets
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train_reg, y_train_reg)

    # Save model
    joblib.dump(lr, "./models/regression/linear_regression.pkl")

    print("Linear Regression training complete!")

if __name__ == "__main__":
    train_linear_regression()
# ======================================================================================================================










# 2==================================================================================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

def train_bgd_regression(X_train, y_train, max_iter=1000, tol=1e-3, learning_rate=0.01):
    """Train Linear Regression using Batch Gradient Descent with SGDRegressor"""
    print("Training Batch Gradient Descent...")
    model = SGDRegressor(
        max_iter=max_iter,
        tol=tol,
        random_state=42,
        learning_rate='constant',
        eta0=learning_rate,
        penalty=None,  # No regularization for pure gradient descent
        shuffle=False  # Don't shuffle for true batch gradient descent
    )
    model.fit(X_train, y_train)
    return model

def train_sgd_regression(X_train, y_train, max_iter=1000, tol=1e-3):
    """Train Linear Regression using Stochastic Gradient Descent"""
    print("Training Stochastic Gradient Descent...")
    model = SGDRegressor(
        max_iter=max_iter,
        tol=tol,
        random_state=42,
        learning_rate='optimal',  # Better learning rate schedule for SGD
        penalty='l2',  # Add some regularization for SGD
        alpha=0.0001,
        shuffle=True  # Shuffle data for true stochastic behavior
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Print coefficients for comparison
    if hasattr(model, 'coef_'):
        print(f"Number of coefficients: {len(model.coef_)}")
        print(f"Intercept: {model.intercept_:.4f}")
    
    return {
        'model_name': model_name,
        'mse': mse,
        'r2': r2,
        'predictions': y_pred,
        'model': model
    }

def train_linear_regression_models():
    """Train three different linear regression approaches"""
    print("Loading processed regression data...")
    
    try:
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
        
        # Ensure models directory exists
        os.makedirs("models/regression", exist_ok=True)
        
        # Train OLS Regression (Normal Equation)
        print("\n" + "="*50)
        ols_model = train_ols_regression(X_train, y_train)
        ols_results = evaluate_model(ols_model, X_test, y_test, "OLS Regression")
        
        # # Train Batch Gradient Descent
        # print("\n" + "="*50)
        # bgd_model = train_bgd_regression(X_train, y_train)
        # bgd_results = evaluate_model(bgd_model, X_test, y_test, "Batch Gradient Descent")
        
        # # Train Stochastic Gradient Descent
        # print("\n" + "="*50)
        # sgd_model = train_sgd_regression(X_train, y_train)
        # sgd_results = evaluate_model(sgd_model, X_test, y_test, "Stochastic Gradient Descent")
        
        # Save all models
        print("\nSaving models...")
        joblib.dump(ols_model, "models/regression/ols_model.pkl")
        # joblib.dump(bgd_model, "models/regression/bgd_model.pkl")
        # joblib.dump(sgd_model, "models/regression/sgd_model.pkl")
        
        # Compare results
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        results = [ols_results, bgd_results, sgd_results]
        for result in results:
            print(f"{result['model_name']:30} | MSE: {result['mse']:8.4f} | R²: {result['r2']:6.4f}")
        
        # Find best model
        best_model = min(results, key=lambda x: x['mse'])
        print(f"\nBest model: {best_model['model_name']} (MSE: {best_model['mse']:.4f})")
        
        print("\nAll models trained and saved successfully!")
        return results
        
    except FileNotFoundError:
        print("Error: Processed data not found. Please run preprocessing first.")
        return None
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

if __name__ == "__main__":
    train_linear_regression_models()
    # ====================================================================================================================================================







# 3================================================================================================================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

def train_linear_regression_gd():
    print("Training Linear Regression using Batch Gradient Descent...")

    # Load your processed dataset
    reg_df = pd.read_csv("data/processed/regression_data.csv")

    # Features and target
    X_reg = reg_df.drop('Temperature (C)', axis=1)
    y_reg = reg_df['Temperature (C)']

    # Train-test split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Pipeline = Standardization + SGDRegressor
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDRegressor(
            loss="squared_error",   # Linear Regression loss
            penalty=None,           # No regularization
            learning_rate="constant",
            eta0=0.001,             # Learning Rate
            max_iter=1000,          # Epochs
            tol=1e-6
        ))
    ])

    # Train model using Gradient Descent
    model.fit(X_train_reg, y_train_reg)

    # Save model
    joblib.dump(model, "./models/regression/linear_regression_gd.pkl")

    print("Batch Gradient Descent training complete!")

if __name__ == "__main__":
    train_linear_regression_gd()



# =====================================================================================================







# 4===============================================================================================================================================================



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_models():
    # Regression models
    print("Training regression models...")
    reg_df = pd.read_csv("data/processed/regression_data.csv")
    X_reg = reg_df.drop('Temperature (C)', axis=1)
    y_reg = reg_df['Temperature (C)']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_reg, y_train_reg)
    joblib.dump(lr, "./models/regression/linear_regression_model.pkl")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    train_models()





# ==========================================================================================================================================================






#5 ============================================


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

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

def train_sgd_regression(X_train, y_train, max_iter=1000, tol=1e-3):
    """Train Linear Regression using Stochastic Gradient Descent"""
    print("Training Stochastic Gradient Descent...")
    model = SGDRegressor(
        max_iter=max_iter,
        tol=tol,
        random_state=42,
        learning_rate='optimal',  # Better learning rate schedule for SGD
        penalty='l2',  # Add some regularization for SGD
        alpha=0.0001,
        shuffle=True  # Shuffle data for true stochastic behavior
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Print coefficients for comparison
    coef = None
    inter = None

    if hasattr(model, "named_steps"):  # Pipeline case
        coef = model.named_steps["sgd"].coef_
        inter = model.named_steps["sgd"].intercept_
    elif hasattr(model, "coef_"):      # OLS case
        coef = model.coef_
        inter = model.intercept_

    if coef is not None:
        print(f"Number of coefficients: {len(coef)}")
        print(f"Intercept: {inter}")

    
    return {
        'model_name': model_name,
        'mse': mse,
        'r2': r2,
        'predictions': y_pred,
        'model': model
    }

def train_linear_regression_models():
    """Train three different linear regression approaches"""
    print("Loading processed regression data...")
    
    try:
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
        
        # Ensure models directory exists
        os.makedirs("models/regression", exist_ok=True)
        
        # Train OLS Regression (Normal Equation)
        print("\n" + "="*50)
        ols_model = train_ols_regression(X_train, y_train)
        ols_results = evaluate_model(ols_model, X_test, y_test, "OLS Regression")
        
        # Train Batch Gradient Descent
        print("\n" + "="*50)
        bgd_model = train_bgd_regression(X_train, y_train)
        bgd_results = evaluate_model(bgd_model, X_test, y_test, "Batch Gradient Descent")
        
        # # Train Stochastic Gradient Descent
        # print("\n" + "="*50)
        # sgd_model = train_sgd_regression(X_train, y_train)
        # sgd_results = evaluate_model(sgd_model, X_test, y_test, "Stochastic Gradient Descent")
        
        # Save all models
        print("\nSaving models...")
        joblib.dump(ols_model, "models/regression/ols_model.pkl")
        joblib.dump(bgd_model, "models/regression/bgd_model.pkl")
        # joblib.dump(sgd_model, "models/regression/sgd_model.pkl")
        
        # Compare results
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        results = [ols_results, bgd_results]
        for result in results:
            print(f"{result['model_name']:30} | MSE: {result['mse']:8.4f} | R²: {result['r2']:6.4f}")
        
        # Find best model
        best_model = min(results, key=lambda x: x['mse'])
        print(f"\nBest model: {best_model['model_name']} (MSE: {best_model['mse']:.4f})")
        
        print("\nAll models trained and saved successfully!")
        return results
        
    except FileNotFoundError:
        print("Error: Processed data not found. Please run preprocessing first.")
        return None
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

if __name__ == "__main__":
    train_linear_regression_models()






    # ======================================================================================
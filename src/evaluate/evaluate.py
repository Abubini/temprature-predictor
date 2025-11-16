import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_models_and_data():
    """Load trained models and test data"""
    print("Loading trained models and data...")
    
    try:
        # Load models
        ols_model = joblib.load("models/regression/ols_model.pkl")
        bgd_model = joblib.load("models/regression/bgd_model.pkl")
        sgd_model = joblib.load("models/regression/sgd_model.pkl")
        
        # Load processed data to create test set
        reg_df = pd.read_csv("data/processed/regression_data.csv")
        X = reg_df.drop('Temperature (C)', axis=1)
        y = reg_df['Temperature (C)']
        
        # Split data (using same random state as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        models = {
            'ols': ols_model,
            'bgd': bgd_model,
            'sgd': sgd_model
        }
        
        return models, X_test, y_test, X_train, y_train
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure models are trained first by running train.py")
        return None, None, None, None, None

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and return detailed metrics"""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print detailed results
    print(f"\n{model_name} Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
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
        print(f"Intercept: {float(inter):.4f}")
        print("First 5 coefficients:", [f"{c:.4f}" for c in coef[:5]])
    
    return {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred,
        'model': model,
        'actual': y_test
    }

def create_regression_visualizations(models, X_test, y_test, X_train=None, y_train=None):
    """Create comprehensive visualizations for all models"""
    print("\nCreating visualizations...")
    
    # Ensure visuals directory exists
    os.makedirs("visuals/regression", exist_ok=True)
    
    # Create subplots for comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Linear Regression Models - Performance Visualizations', fontsize=16, fontweight='bold')
    
    # Colors for different models
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    model_names = ['OLS Regression', 'Batch Gradient Descent', 'Stochastic Gradient Descent']
    model_keys = ['ols', 'bgd', 'sgd']
    
    # Plot 1: Actual vs Predicted for all models
    for idx, (model_key, model_name, color) in enumerate(zip(model_keys, model_names, colors)):
        model = models[model_key]
        y_pred = model.predict(X_test)
        
        # Scatter plot: Actual vs Predicted
        ax1 = axes[0, idx]
        ax1.scatter(y_test, y_pred, alpha=0.6, color=color, s=50)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Actual Temperature (C)')
        ax1.set_ylabel('Predicted Temperature (C)')
        ax1.set_title(f'{model_name}\nActual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(y_test, y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    # Plot 2: Residual plots
    for idx, (model_key, model_name, color) in enumerate(zip(model_keys, model_names, colors)):
        model = models[model_key]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        ax2 = axes[1, idx]
        ax2.scatter(y_pred, residuals, alpha=0.6, color=color, s=50)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Predicted Temperature (C)')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name}\nResidual Plot')
        ax2.grid(True, alpha=0.3)
        
        # Add residual statistics
        mean_residual = residuals.mean()
        ax2.text(0.05, 0.95, f'Mean Residual: {mean_residual:.4f}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("visuals/regression/all_models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual detailed plots for each model
    for model_key, model_name, color in zip(model_keys, model_names, colors):
        model = models[model_key]
        y_pred = model.predict(X_test)
        
        # Create individual model visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Detailed Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted with regression line
        ax1.scatter(y_test, y_pred, alpha=0.6, color=color, s=50)
        
        # Add regression line for the predictions
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(y_test, p(y_test), "r--", alpha=0.8, linewidth=2, 
                label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'g-', alpha=0.5, linewidth=1, 
                label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Temperature (C)')
        ax1.set_ylabel('Predicted Temperature (C)')
        ax1.set_title('Actual vs Predicted with Best Fit Line')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residual distribution
        residuals = y_test - y_pred
        ax2.hist(residuals, bins=30, alpha=0.7, color=color, edgecolor='black')
        ax2.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {residuals.mean():.3f}')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Prediction error over actual values
        ax3.scatter(y_test, residuals, alpha=0.6, color=color, s=50)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Actual Temperature (C)')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Prediction Error vs Actual Values')
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q plot for normality of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot: Normality of Residuals')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"visuals/regression/{model_key}_detailed_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create coefficients comparison plot (for OLS and feature importance)
    if hasattr(models['ols'], 'coef_'):
        plt.figure(figsize=(12, 8))
        
        # Get feature names
        feature_names = X_test.columns
        
        # Plot OLS coefficients
        coefficients = models['ols'].coef_
        sorted_idx = np.argsort(np.abs(coefficients))[::-1]
        
        plt.barh(range(len(sorted_idx[:10])), 
                coefficients[sorted_idx[:10]], 
                color='skyblue', edgecolor='black')
        plt.yticks(range(len(sorted_idx[:10])), 
                  [feature_names[i] for i in sorted_idx[:10]])
        plt.xlabel('Coefficient Value')
        plt.title('Top 10 Feature Coefficients - OLS Regression')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig("visuals/regression/feature_coefficients.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualizations saved to: visuals/regression/")

def compare_models(results):
    """Compare all models and display results"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': [r['model_name'] for r in results],
        'MSE': [r['mse'] for r in results],
        'RMSE': [r['rmse'] for r in results],
        'MAE': [r['mae'] for r in results],
        'R²': [r['r2'] for r in results]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Find best models by different metrics
    best_mse = min(results, key=lambda x: x['mse'])
    best_r2 = max(results, key=lambda x: x['r2'])
    best_mae = min(results, key=lambda x: x['mae'])
    
    print(f"\nBest model by MSE: {best_mse['model_name']} (MSE: {best_mse['mse']:.4f})")
    print(f"Best model by R²: {best_r2['model_name']} (R²: {best_r2['r2']:.4f})")
    print(f"Best model by MAE: {best_mae['model_name']} (MAE: {best_mae['mae']:.4f})")
    
    return comparison_df

def evaluate_all_models():
    """Main function to evaluate all trained models"""
    print("Starting Model Evaluation...")
    print("="*50)
    
    # Load models and data
    models, X_test, y_test, X_train, y_train = load_models_and_data()
    
    if models is None:
        return None
    
    print(f"Test set shape: {X_test.shape}")
    print(f"Models loaded: {list(models.keys())}")
    
    # Evaluate each model
    results = []
    
    print("\n" + "="*50)
    ols_results = evaluate_model(models['ols'], X_test, y_test, "OLS Regression")
    results.append(ols_results)
    
    print("\n" + "="*50)
    bgd_results = evaluate_model(models['bgd'], X_test, y_test, "Batch Gradient Descent")
    results.append(bgd_results)
    
    print("\n" + "="*50)
    sgd_results = evaluate_model(models['sgd'], X_test, y_test, "Stochastic Gradient Descent")
    results.append(sgd_results)
    
    # Create visualizations
    create_regression_visualizations(models, X_test, y_test, X_train, y_train)
    
    # Compare all models
    comparison_df = compare_models(results)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETED!")
    print("="*50)
    print("Visualizations saved in: visuals/regression/")
    
    return results, comparison_df

if __name__ == "__main__":
    evaluation_results = evaluate_all_models()
    
    if evaluation_results is not None:
        print("\nEvaluation completed successfully!")
        print("Check the 'visuals/regression/' folder for all graphs and plots.")
    else:
        print("\nEvaluation failed!")
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
    
    # # Decision Tree
    # dt = DecisionTreeRegressor(random_state=42)
    # dt.fit(X_train_reg, y_train_reg)
    # joblib.dump(dt, "./models/regression/decision_tree_model.pkl")
    
    # # Classification models
    # print("\nTraining classification models...")
    # clf_df = pd.read_csv("./data/processed/classification_data.csv")
    # X_clf = clf_df.drop(['Summary', 'Precip Type', 'Summary_encoded', 'Precip_encoded'], axis=1)
    # y_clf_summary = clf_df['Summary_encoded']
    # y_clf_precip = clf_df['Precip_encoded']
    # X_train_clf, X_test_clf, y_train_summary, y_test_summary, y_train_precip, y_test_precip = train_test_split(
    #     X_clf, y_clf_summary, y_clf_precip, test_size=0.2, random_state=42
    # )
    
    # # Random Forest for Summary
    # rf_summary = RandomForestClassifier(random_state=42)
    # rf_summary.fit(X_train_clf, y_train_summary)
    # joblib.dump(rf_summary, "models/classification/random_forest_summary.pkl")
    
    # # XGBoost for Precipitation
    # xgb_precip = XGBClassifier(random_state=42)
    # xgb_precip.fit(X_train_clf, y_train_precip)
    # joblib.dump(xgb_precip, "models/classification/xgboost_precip.pkl")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    train_models()
import joblib
import pandas as pd

class TemperaturePredictor:
    def __init__(self):
        self.models = {
            'ols': joblib.load("models/regression/ols_model.pkl"),
            'bgd': joblib.load("models/regression/bgd_model.pkl"),
            'sgd': joblib.load("models/regression/sgd_model.pkl"),
        }
    
    def predict_temperature(self, features):
        """Predict temperature using ALL models"""
        input_df = pd.DataFrame([features])
        
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(input_df)[0]
            predictions[name] = pred
        
        return predictions


if __name__ == "__main__":
    predictor = TemperaturePredictor()

    example_features = {
        'Apparent Temperature (C)': 7.27777777,
        'Humidity': 0.86,
        'Wind Speed (km/h)': 14.2646,
        'Wind Bearing (degrees)': 259.0,
        'Visibility (km)': 15.8263,
        'Pressure (millibars)': 1015.63,
        'month': 3,
        'hour': 23
    }
    
    results = predictor.predict_temperature(example_features)

    print("\n=== Predictions from all models ===")
    for model_name, value in results.items():
        print(f"{model_name.upper()} → {value:.2f}°C")

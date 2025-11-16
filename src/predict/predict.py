import joblib
import pandas as pd

class TemperaturePredictor:
    def __init__(self):
        self.models = {
            'linear_regression': joblib.load("models/regression/linear_regression_sgd.pkl"),
            
        }
    
    def predict_temperature(self, features, model_type='linear_regression'):
        """Predict temperature using specified model"""
        model = self.models.get(model_type)
        if not model:
            raise ValueError(f"Model {model_type} not found")
        
        input_df = pd.DataFrame([features])
        return model.predict(input_df)[0]

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
    
    temp = predictor.predict_temperature(example_features, 'linear_regression')
    print(f"Predicted Temp ({'linear_regression'}): {temp:.1f}°C")
"""
Prediction module for real-time and custom temperature prediction.
Contains:
- Real-time API-based prediction (predict.py)
- TemperaturePredictor class for manual feature input (test-predict.py)
"""

from .predict import predict_realtime_temperature, get_realtime_weather, convert_to_features
from .test_predict import TemperaturePredictor

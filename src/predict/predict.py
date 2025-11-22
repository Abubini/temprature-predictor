import requests
import joblib
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import os



# -----------------------------
# 1. Load your trained models
# -----------------------------
MODEL_OLS = "models/regression/ols_model.pkl"
MODEL_BGD = "models/regression/bgd_model.pkl"
MODEL_SGD = "models/regression/sgd_model.pkl"

models = {
    "OLS Regression": joblib.load(MODEL_OLS),
    "Batch Gradient Descent": joblib.load(MODEL_BGD),
    "Stochastic Gradient Descent": joblib.load(MODEL_SGD)
}

# -----------------------------
# 2. OpenWeatherMap API Key
# -----------------------------
load_dotenv()

API_KEY = os.getenv("OWM_API_KEY")

# -----------------------------
# 3. Fetch real-time weather with error handling
# -----------------------------
def get_realtime_weather(lat=9.03, lon=38.74):
    """
    Fetch current weather data from OpenWeatherMap.
    Returns None if API fails or key is invalid.
    """
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        print("Error connecting to OpenWeatherMap:", e)
        return None

    # Check for API errors
    if "main" not in data:
        print("Error fetching weather data:", data)
        return None

    # Extract features
    weather = {
        "temperature": data["main"]["temp"],                      # Actual temperature
        "humidity": data["main"]["humidity"] / 100,               # 0-1
        "pressure": data["main"]["pressure"],                     # millibars
        "visibility": data.get("visibility", 10000) / 1000,       # meters → km
        "wind_speed": data["wind"].get("speed", 0) * 3.6,         # m/s → km/h
        "wind_bearing": data["wind"].get("deg", 0),               # degrees
        "actual_temp": data["main"]["temp"]                       # For comparison
    }

    return weather

# -----------------------------
# 4. Convert API data -> model features
# -----------------------------
def convert_to_features(w):
    now = datetime.now()
    return np.array([[
        w["temperature"],    # Apparent temperature (C)
        w["humidity"],       # 0-1
        w["wind_speed"],     # km/h
        w["wind_bearing"],   # degrees
        w["visibility"],     # km
        w["pressure"],       # millibars
        now.month,
        now.hour
    ]])

# -----------------------------
# 5. Predict temperature (ALL MODELS)
# -----------------------------
def predict_realtime_temperature():
    print("\nFetching real-time weather from OpenWeatherMap...")
    rt = get_realtime_weather()

    if rt is None:
        print("Cannot predict because weather fetch failed.")
        return

    print("\nReal-time weather received:")
    for k, v in rt.items():
        print(f"  {k}: {v}")

    X = convert_to_features(rt)

    print("\n==============================")
    # Predict with every model
    for model_name, model in models.items():
        pred = model.predict(X)[0]
        print(f"{model_name} Prediction (C): {pred:.4f}")
        print(f"Actual Temp from OWM (C): {rt['actual_temp']:.4f}")

    print("==============================\n")

    return True

# -----------------------------
# Run script directly
# -----------------------------
if __name__ == "__main__":
    predict_realtime_temperature()

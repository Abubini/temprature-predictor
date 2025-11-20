import requests
import joblib
import numpy as np
from datetime import datetime

# -----------------------------
# 1. Load your trained model
# -----------------------------
MODEL_PATH = "models/regression/ols_model.pkl"
model = joblib.load(MODEL_PATH)

# -----------------------------
# 2. OpenWeatherMap API Key
# -----------------------------
API_KEY = "d0aa7f482997759c0ab527e9f7f6e250"

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
# 5. Predict temperature
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
    predicted_temp = model.predict(X)[0]

    print("\n==============================")
    print(f"Predicted Temperature (C): {predicted_temp:.2f}")
    print(f"Actual Temp from OWM (C): {rt['actual_temp']:.2f}")
    print(f"Difference: {abs(predicted_temp - rt['actual_temp']):.2f}°C")
    print("==============================\n")

    return predicted_temp

# -----------------------------
# Run script directly
# -----------------------------
if __name__ == "__main__":
    predict_realtime_temperature()

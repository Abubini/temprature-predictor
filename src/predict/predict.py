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
def get_realtime_weather():
    import requests
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=Addis+Ababa&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            print("Weather API Error:", data)
            return None
        return {
            "actual_temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"] / 100,
            "wind_speed": data["wind"]["speed"],
            "wind_bearing": data["wind"]["deg"],
            "pressure": data["main"]["pressure"],
            "visibility": data.get("visibility", 10000)/1000,
            "condition": data["weather"][0]["main"],
            "location": data["name"],
            "country": data["sys"]["country"]
        }
    except Exception as e:
        print("Exception in get_realtime_weather:", e)
        return None

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

# Weather Temperature Prediction using Linear Regression

## 📋 Project Overview

This project implements and compares three different linear regression approaches to predict temperature based on weather features using historical weather data. The system includes data preprocessing, model training, evaluation, and real-time prediction capabilities.

---

## 🎯 Project Goals

* Implement and compare OLS, Batch Gradient Descent, and Stochastic Gradient Descent for linear regression
* Build a robust pipeline for weather data preprocessing and feature engineering
* Create comprehensive model evaluation and visualization
* Develop real-time temperature prediction using OpenWeatherMap API

---

## 👥 Development Team

* Biniyam Girma (ATE/7146/14)
* Simon Mesfin (7211/14)
* Yosef Ashebir (ATE4638/14)

---

## 🏗️ Project Architecture

```
weather-regression/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned and feature-engineered data
├── models/
│   └── regression/          # Trained model files
├── visuals/
│   └── regression/          # Evaluation plots and charts
├── src/                     # Source code
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── train.py           # Model training implementation
│   ├── evaluate.py        # Model evaluation and visualization
│   └── predict.py         # Real-time prediction
└── README.md
```

---

## 📊 Dataset Information

**Source**

* Dataset: Historical Weather Data
* File: `weatherHistory.csv`
* Location: Various weather stations
* Time Period: Historical records with timestamps

### Features Used for Regression

| Feature                  |           Description |        Type | Transformation |
| ------------------------ | --------------------: | ----------: | -------------- |
| Apparent Temperature (C) | Perceived temperature |   Numerical | Direct use     |
| Humidity                 |    Air humidity level |   Numerical | 0-1 scale      |
| Wind Speed (km/h)        |         Wind velocity |   Numerical | Direct use     |
| Wind Bearing (degrees)   |        Wind direction |   Numerical | Direct use     |
| Visibility (km)          |   Visibility distance |   Numerical | Direct use     |
| Pressure (millibars)     |  Atmospheric pressure |   Numerical | Direct use     |
| Month                    |      Temporal feature | Categorical | 1-12 encoding  |
| Hour                     |      Temporal feature | Categorical | 0-23 encoding  |

**Target Variable**

* Temperature (C): Actual measured temperature in Celsius

---

## 🧮 Implemented Algorithms

### 1. Ordinary Least Squares (OLS) Regression

* **Method:** Normal Equation (Closed-form solution)
* **Advantages:** Exact solution, no hyperparameters
* **Implementation:** `sklearn.linear_model.LinearRegression`
* **Use Case:** Baseline model for comparison

### 2. Batch Gradient Descent (BGD)

* **Method:** Iterative optimization using entire dataset
* **Learning Rate:** Constant (0.001)
* **Convergence:** Tolerance-based early stopping (`1e-6`)
* **Preprocessing:** `StandardScaler` for feature normalization
* **Implementation:** `SGDRegressor` with `shuffle=False`

### 3. Stochastic Gradient Descent (SGD)

* **Method:** Iterative optimization using random samples
* **Learning Rate:** Adaptive (initial `0.01`)
* **Shuffling:** Enabled for true stochastic behavior
* **Preprocessing:** `StandardScaler` for feature normalization
* **Implementation:** `SGDRegressor` with `shuffle=True`

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.8+
* `pip` package manager

### Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib requests
```

### Project Structure Setup

```bash
# Create directory structure
mkdir -p data/raw data/processed models/regression visuals/regression src

# Place your dataset
cp weatherHistory.csv data/raw/
```

---

## 💻 Usage Instructions

### Step 1: Data Preprocessing

```bash
cd src
python preprocess.py
```

* **Output:** Creates `data/processed/regression_data.csv` with cleaned and feature-engineered data.

### Step 2: Model Training

```bash
python train.py
```

* **Output:** Trains three models and saves them to `models/regression/`:

  * `ols_model.pkl` - OLS Regression
  * `bgd_model.pkl` - Batch Gradient Descent
  * `sgd_model.pkl` - Stochastic Gradient Descent

### Step 3: Model Evaluation

```bash
python evaluate.py
```

* **Output:**

  * Comprehensive performance metrics
  * Comparison charts in `visuals/regression/`
  * Residual analysis plots

### Step 4: Real-time Prediction

```bash
python predict.py
```

* **Output:** Real-time temperature predictions using current weather data from OpenWeatherMap API.

---

## 📈 Model Evaluation Metrics

**Performance Metrics Used**

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

**Expected Performance**

* R² Score: `> 0.85`
* RMSE: `< 2.0°C`
* MAE: `< 1.5°C`

---

## 🔍 Technical Implementation Details

### Data Preprocessing Pipeline

* Missing Value Handling: Removal of rows with missing precipitation type
* Feature Engineering:

  * DateTime parsing and feature extraction
  * Temporal features (month, hour, day_of_year)
  * Numerical feature scaling
* Feature Selection: Curated set of 8 meteorological features

### Model Training Configuration

```python
# OLS Regression - No hyperparameters
LinearRegression()

# Batch Gradient Descent
SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.001,
    max_iter=1000,
    tol=1e-6,
    shuffle=False
)

# Stochastic Gradient Descent  
SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="adaptive",
    eta0=0.01,
    max_iter=1000,
    shuffle=True
)
```

### Real-time Prediction System

* **API Integration:** OpenWeatherMap REST API
* **Data Transformation:** Automatic feature conversion
* **Multi-model Inference:** Parallel predictions from all trained models
* **Error Handling:** Robust API failure management

---

## 📊 Visualization Outputs

The evaluation script generates comprehensive visualizations:

1. **Model Comparison Plot**

   * File: `all_models_comparison.png`
   * Content: Side-by-side Actual vs Predicted plots for all three models
   * Features: Best-fit lines and perfect prediction reference

2. **Individual Model Analysis**

   * Files: `ols_detailed_analysis.png`, `bgd_detailed_analysis.png`, `sgd_detailed_analysis.png`
   * Content: Actual vs Predicted with regression line, Residual distribution histogram, Model coefficients and intercepts

---

## 🔧 Configuration

### OpenWeatherMap API

* **API Key:** Required for real-time predictions
* **Default Location:** Addis Ababa (Lat: 9.03, Lon: 38.74)
* **Units:** Metric (Celsius, km/h, millibars)

### Training Parameters

* Test Size: 20% holdout for evaluation
* Random State: 42 for reproducible splits
* Validation: Train-test split with stratification

---

## 🐛 Troubleshooting

### Common Issues

**File Not Found Errors**

```bash
# Ensure proper directory structure
mkdir -p data/raw models/regression visuals/regression
```

**API Connection Issues**

* Verify internet connectivity
* Check OpenWeatherMap API key validity
* Confirm API rate limits not exceeded

**Model Training Failures**

* Verify data preprocessing completed successfully
* Check for sufficient memory during training
* Ensure all required features are present

### Debug Mode

```python
import traceback
try:
    # Your code
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
```

---

## 📝 Future Enhancements

### Planned Improvements

* Advanced Feature Engineering

  * Polynomial features for non-linear relationships
  * Interaction terms between meteorological variables
  * Seasonal decomposition features

* Model Enhancements

  * Regularized regression variants (Ridge, Lasso)
  * Ensemble methods combining all approaches
  * Hyperparameter optimization with cross-validation

* System Improvements

  * Database integration for historical data storage
  * Web interface for real-time predictions
  * Automated model retraining pipeline

* Additional Features

  * Weather forecast integration
  * Multiple location support
  * Prediction confidence intervals

---

## 🤝 Contributing

### Development Workflow

* Fork the repository
* Create feature branch (`git checkout -b feature/AmazingFeature`)
* Commit changes (`git commit -m 'Add AmazingFeature'`)
* Push to branch (`git push origin feature/AmazingFeature`)
* Open Pull Request

### Code Standards

* Follow PEP 8 style guide
* Include docstrings for all functions
* Add unit tests for new functionality
* Update documentation accordingly

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 🎓 Academic Reference

This implementation demonstrates practical applications of:

* Linear regression theory and implementations
* Gradient descent optimization algorithms
* Machine learning pipeline development
* API integration for real-time systems
* Model evaluation and visualization techniques

Developed with ❤️ by Biniyam Girma, Simon Mesfin, and Yosef Ashebir

*For questions or support, please contact the development team.*

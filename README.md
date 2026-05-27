# 🌾 Hybrid Crop Prediction System

Machine Learning-based Crop Recommendation System using XGBoost, MLP Neural Network, Live Weather API Integration, and SHAP Explainability.

---

# 📌 Overview

This project is a Flask-based machine learning web application that recommends suitable crops based on:

* Soil nutrients (N, P, K)
* Temperature
* Humidity
* pH value
* Rainfall
* Live weather data fetched using Open-Meteo APIs

The system compares predictions from:

* XGBoost Classifier
* MLPClassifier (Neural Network)

To improve model transparency, SHAP explainability is used to analyze how input features influence predictions.

The application also includes an interactive frontend for manual input and live weather-based prediction.

---

# 🚀 Features

## 🔮 Machine Learning

* Crop recommendation using XGBoost
* Neural network prediction using MLPClassifier
* Feature scaling using StandardScaler
* Label encoding for crop classification
* Train-test split and model evaluation

## 🌤 Live Weather Integration

* Fetches weather data using Open-Meteo APIs
* Retrieves:

  * Temperature
  * Humidity
  * Rainfall estimates

## 📊 Explainable AI

* SHAP explainability integration
* Displays feature contribution analysis
* Helps interpret model predictions

## 💻 Web Application

* Flask backend
* Interactive frontend using HTML, CSS, and JavaScript
* Manual input mode
* Weather API-based input mode

---

# 🧩 Project Structure

```bash
Crop-Prediction-System/
│
├── app_flask.py                  # Flask application
├── exe.py                        # Alternate Flask runner
├── harCode.py                    # Model training and prediction logic
├── index.html                    # Frontend UI
├── Crop_recommendation-1.csv     # Dataset
└── README.md
```

---

# 📦 Technologies Used

## Backend & Machine Learning

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Requests

## Frontend

* HTML
* CSS
* JavaScript

## APIs

* Open-Meteo Geocoding API
* Open-Meteo Forecast API

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd Crop-Prediction-System
```

## 2️⃣ Install Dependencies

```bash
pip install flask pandas numpy scikit-learn xgboost shap requests
```

## 3️⃣ Run the Application

```bash
python app_flask.py
```

## 4️⃣ Open in Browser

```bash
http://127.0.0.1:5000/
```

---

# 📁 Dataset Description

The dataset contains agricultural and environmental parameters used for crop recommendation.

| Feature     | Description        |
| ----------- | ------------------ |
| N           | Nitrogen content   |
| P           | Phosphorus content |
| K           | Potassium content  |
| temperature | Temperature in °C  |
| humidity    | Relative humidity  |
| ph          | Soil pH            |
| rainfall    | Rainfall in mm     |
| label       | Recommended crop   |

---

# 🤖 Machine Learning Workflow

## Data Preprocessing

* Dataset loading using Pandas
* Feature-label separation
* Label encoding
* Feature scaling using StandardScaler
* Train-test split

## Models Used

### XGBoost Classifier

* Used for structured crop prediction
* Handles tabular data efficiently

### MLPClassifier

* Feedforward neural network model
* Used for comparative prediction analysis

## Prediction Flow

User Input → Feature Scaling → Model Prediction → SHAP Explainability → Result Display

---

# 🌤 Weather API Integration

The system uses Open-Meteo APIs to automatically fetch weather information based on location input.

## Weather Data Retrieved

* Temperature
* Humidity
* Rainfall

---

# 📊 SHAP Explainability

SHAP values are used to explain how each feature contributes to the prediction.

This helps users understand:

* Which features positively influence predictions
* Which features negatively influence predictions

---

# 🛠 API Endpoints

## POST /fetch_weather

### Request

```json
{
  "location": "Hyderabad"
}
```

### Response

```json
{
  "success": true,
  "temperature": 28.5,
  "humidity": 40,
  "rainfall": 64.0
}
```

---

## POST /predict

### Request

```json
{
  "N": 50,
  "P": 40,
  "K": 35,
  "temperature": 25,
  "humidity": 60,
  "ph": 6.5,
  "rainfall": 120
}
```

### Response

```json
{
  "success": true,
  "xgboost": {
    "crop": "rice"
  },
  "mlp": {
    "crop": "rice"
  }
}
```

---

# 📘 Future Improvements

* Add additional machine learning models for comparison
* Deploy using cloud platforms
* Improve UI responsiveness
* Integrate real-time agricultural datasets

---
## 👩‍💻 Author

### Harshitha Mekala

Recent Computer Science Graduate passionate about Python, SQL, Power BI, Machine Learning, and Data Analytics.

🔗 GitHub: https://github.com/Harshitha152  
🔗 LinkedIn: https://www.linkedin.com/in/harshitha-mekala15

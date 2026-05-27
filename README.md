<h1 align="center">🌾 Hybrid Crop Prediction System</h1>

<p align="center">
Hybrid AI | XGBoost | MLPClassifier | SHAP Explainability
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge"/>
<img src="https://img.shields.io/badge/MLPClassifier-6A1B9A?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SHAP_Explainability-FFCC00?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Hybrid_AI-102230?style=for-the-badge"/>
<img src="https://img.shields.io/badge/API_Integration-0A66C2?style=for-the-badge"/>
</p>

---

## 📌 Overview

This project is a Hybrid Crop Prediction System that recommends suitable crops using agricultural and environmental parameters.

The implementation uses:

* XGBoost Classifier for crop prediction
* MLPClassifier for comparative prediction analysis
* SHAP Explainability for feature contribution analysis
* Open-Meteo APIs for live weather data integration

The system analyzes:

* Soil nutrients (N, P, K)
* Temperature
* Humidity
* pH values
* Rainfall data

to generate crop predictions and interpret model behavior.

---

## 🚀 Features

✅ Hybrid Crop Prediction System
✅ XGBoost-Based Crop Recommendation
✅ MLPClassifier Prediction Analysis
✅ SHAP Explainability Integration
✅ Feature Contribution Analysis
✅ Live Weather API Integration
✅ Feature Scaling & Label Encoding
✅ Weather-Based Prediction Input

---

## 🛠️ Tech Stack

### Machine Learning & Data Processing

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP

### API Integration

* Requests
* Open-Meteo APIs

### Frontend

* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```bash id="0prnlr"
Hybrid-Crop-Prediction-System/
│── app_flask.py
│── exe.py
│── harCode.py
│── index.html
│── Crop_recommendation-1.csv
│── README.md
```

---

## 📁 Dataset Description

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

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash id="o8jfln"
git clone <repository-url>
cd Hybrid-Crop-Prediction-System
```

### 2️⃣ Install Dependencies

```bash id="00pqco"
pip install pandas numpy scikit-learn xgboost shap requests flask
```

### 3️⃣ Run the Application

```bash id="mx7v4g"
python app_flask.py
```

---

## 🤖 Prediction Workflow

Input Data
↓
Preprocessing
↓
Feature Scaling
↓
XGBoost Prediction
↓
MLPClassifier Prediction
↓
SHAP Explainability
↓
Result Analysis

---

## 🌤️ Weather API Integration

The system uses Open-Meteo APIs to fetch:

* Temperature
* Humidity
* Rainfall information

based on user location input.

---

## 📊 SHAP Explainability

SHAP values are used to analyze feature contribution for crop predictions.

This helps interpret:

* positive feature influence
* negative feature influence
* prediction behavior

---

## 🛠️ API Endpoints

### POST /fetch_weather

```json id="zb8u5m"
{
  "location": "Hyderabad"
}
```

---

### POST /predict

```json id="sllvqm"
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

---

## 📘 Future Improvements

* Add additional model comparisons
* Improve prediction analysis
* Add deployment support
* Integrate larger agricultural datasets

---

## 👩‍💻 Author

### Harshitha Mekala

Computer Science Engineering Graduate specializing in Data Science with interests in Machine Learning, Data Analytics, and AI-based Systems.

📧 [harshithamekala15@gmail.com](mailto:harshithamekala15@gmail.com)

🔗 GitHub: https://github.com/Harshitha152
🔗 LinkedIn: https://www.linkedin.com/in/harshitha-mekala15

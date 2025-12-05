🌾 Hybrid Crop Prediction System

AI-Powered Crop Recommendation using XGBoost, Deep Neural Network (MLP), Live Weather API & SHAP Explainability

📌 Overview

The Hybrid Crop Prediction System is a full-stack machine-learning application that recommends the most suitable crop based on:

Soil nutrients (N, P, K)

pH value

Temperature

Humidity

Rainfall

Live weather data fetched automatically from the Open-Meteo API

The system uses a hybrid modelling approach combining:

XGBoost Classifier

Deep Neural Network (MLPClassifier)

To increase transparency, the system also includes:

SHAP Explainability

Feature-wise impact visualization (positive/negative influence on prediction)

Model accuracy display

The front-end is an elegant interactive dashboard allowing users to enter values manually or fetch live weather data by city.

🚀 Features
🔮 Machine Learning

XGBoost & MLP models trained on the Crop Recommendation Dataset

StandardScaler for feature normalization

LabelEncoder for multi-class crop labels

Train/test split with accuracy reporting

SHAP explainers for feature impact analysis

🌤 Live Weather Integration

City-based weather fetching using:

Open-Meteo Geocoding API

Open-Meteo Forecast API

Automatically retrieves:
✔ Temperature
✔ Relative humidity
✔ Rainfall estimate

📊 Explainable AI

SHAP values extracted for each model

Feature-level analysis rendered as dynamic charts

Shows how each feature pushes the model toward or away from the predicted crop

💻 Front-End UI

Two input modes:
✔ Weather Fetch Mode
✔ Manual Entry Mode

Animated cards, tabs, alerts, loaders, and responsive design

Real-time prediction results with SHAP bar charts

🧩 Backend API (Flask)

/ — main UI

/fetch_weather — fetch live weather

/predict — predict crop using both models

🏗 Project Structure
📁 Crop-Prediction-System/
│
├── app_flask.py               # Main Flask application
├── exe.py                     # Alternate Flask runner (same functionality)
├── harCode.py                 # ML training, SHAP logics, weather API calls
├── index.html                 # Complete front-end UI
├── Crop_recommendation-1.csv  # Dataset used for model training
└── harCode.cpython-313.pyc    # Compiled Python file (ignored)

📦 Technologies Used
Backend / ML

Python

Flask

Pandas, NumPy

Scikit-learn

XGBoost

SHAP

Requests

Frontend

HTML5, CSS3

JavaScript (Vanilla)

Animated UI components

External APIs

Open-Meteo Geocoding API

Open-Meteo Weather Forecast API

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone <repo-url>
cd Crop-Prediction-System

2️⃣ Install Dependencies
pip install -r requirements.txt


(If you don’t have a requirements.txt, create one using the list below):

flask
pandas
numpy
scikit-learn
xgboost
shap
requests

3️⃣ Run the Application
python app_flask.py

4️⃣ Open in Browser
http://127.0.0.1:5000/

📁 Dataset Description

The project uses Crop_recommendation-1.csv, which contains:

Feature	Description
N	Nitrogen content (mg/kg)
P	Phosphorus content (mg/kg)
K	Potassium content (mg/kg)
temperature	Air temperature in °C
humidity	Relative humidity (%)
ph	Soil pH (0–14)
rainfall	Rainfall in mm
label	Recommended crop
🤖 Machine Learning Workflow
1. Data Preprocessing

Load CSV

Split into features (X) and labels (y)

Encode labels

Normalize features

2. Model Training
⭐ XGBoost

100 estimators

Trained on scaled features

Uses Tree SHAP explainer

🧠 MLP (Neural Network)

2 hidden layers (64, 32)

Max iter: 300

KernelExplainer for SHAP

3. Prediction

User input → scaled → fed into both models

Outputs: crop, confidence score

SHAP values generated

🌈 Front-End UI Highlights
Weather Tab

Enter city → fetch live weather

Auto-fill temp/humidity/rainfall

Combine with soil nutrients

Manual Tab

Enter all parameters manually

Results Section

XGBoost & MLP predictions side-by-side

Confidence score

SHAP bar graphs for:

Positive impact (green)

Negative impact (red)

🛠 API Endpoints
POST /fetch_weather

Request:

{
  "location": "Hyderabad"
}


Response:

{
  "success": true,
  "temperature": 28.5,
  "humidity": 40,
  "rainfall": 64.0
}

POST /predict

Request:

{
  "N": 50,
  "P": 40,
  "K": 35,
  "temperature": 25,
  "humidity": 60,
  "ph": 6.5,
  "rainfall": 120
}


Response:

{
  "success": true,
  "xgboost": { "crop": "rice", "confidence": 89.3, "shap": [...] },
  "mlp": { "crop": "rice", "confidence": 81.5, "shap": [...] }
}

📊 SHAP Explainability

The system explains "why" a crop was recommended:

Each feature gets a SHAP value

Positive SHAP → pushes prediction towards crop

Negative SHAP → pushes prediction away

Values normalized and displayed visually

This promotes trust, transparency, and interpretability.

📘 Future Enhancements

Add rainfall & soil data via government APIs

Add RandomForest/SVM for comparison

Deploy using Docker / Render / HuggingFace Spaces

Add user login & prediction history

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import requests
import shap
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

def fetch_live_weather(location):
    print(f"Fetching weather for {location} ...")
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_response = requests.get(geo_url, timeout=10)
        geo_data = geo_response.json()
        if 'results' not in geo_data or len(geo_data['results']) == 0:
            print(f"❌ Location '{location}' not found")
            return None
        latitude = geo_data['results'][0]['latitude']
        longitude = geo_data['results'][0]['longitude']
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        weather_response = requests.get(weather_url, timeout=10)
        weather_data = weather_response.json()
        temperature = weather_data['current']['temperature_2m']
        humidity = weather_data['current']['relative_humidity_2m']
        precipitation = weather_data['current'].get('precipitation', 0)
        rainfall = max(precipitation * 30, 20)
        return {
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'latitude': latitude,
            'longitude': longitude
        }
    except Exception as e:
        print(f"❌ Error fetching live weather: {e}")
        return None

def get_manual_input():
    """Allow user to manually input all parameters"""
    print("🌱 Manual Input Mode - Enter all parameters:")
    print("=" * 50)
    
    try:
        N = float(input('N (Nitrogen, mg/kg): '))
        P = float(input('P (Phosphorus, mg/kg): '))
        K = float(input('K (Potassium, mg/kg): '))
        temperature = float(input('Temperature (°C): '))
        humidity = float(input('Humidity (%): '))
        ph = float(input('pH (0-14): '))
        rainfall = float(input('Rainfall (mm): '))
        
        return {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
    except ValueError:
        print("❌ Invalid input! Please enter numeric values only.")
        return None

def get_input_data(X):
    """Get input data from user with option selection"""
    print("🎯 Choose input method:")
    print("1. 🌤️  Weather Fetching (Get live weather data)")
    print("2. ✏️  Manual Input (Enter all parameters manually)")
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice not in ['1', '2']:
                print("❌ Please enter 1 or 2")
                continue
            
            if choice == '1':
                # Weather fetching option
                location = input('Enter city: ')
                weather = fetch_live_weather(location)
                if not weather:
                    print("❌ Could not fetch weather data. Please try manual input.")
                    continue
                
                print("Enter soil nutrients:")
                N = float(input('N: ')); P = float(input('P: ')); K = float(input('K: ')); ph = float(input('pH: '))
                
                input_data = pd.DataFrame([[N, P, K, weather['temperature'], weather['humidity'], ph, weather['rainfall']]],
                                         columns=X.columns)
                print(f"✅ Using weather data: Temp={weather['temperature']}°C, Humidity={weather['humidity']}%, Rainfall={weather['rainfall']}mm")
                
            else:
                # Manual input option
                manual_data = get_manual_input()
                if not manual_data:
                    print("❌ Invalid input. Please try again.")
                    continue
                
                input_data = pd.DataFrame([[manual_data['N'], manual_data['P'], manual_data['K'], 
                                          manual_data['temperature'], manual_data['humidity'], 
                                          manual_data['ph'], manual_data['rainfall']]], 
                                         columns=X.columns)
                print("✅ Using manual input data")
            
            return input_data
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values only.")
        except Exception as e:
            print(f"❌ Error: {e}")

def explain_with_shap(model, explainer, input_df, label_encoder, model_name):
    # ensure 2D float64 input
    input_array = np.ascontiguousarray(input_df.to_numpy(dtype=np.float64, copy=True))
    pred = model.predict(input_array)
    proba = model.predict_proba(input_array)[0]
    class_idx = int(np.argmax(proba))
    pred_crop = label_encoder.inverse_transform([pred[0]])[0]

    print(f'--- {model_name} Prediction ---')
    print(f'🌾 Crop: {pred_crop}   Confidence: {proba[class_idx]*100:.2f}%')

    try:
        sv = explainer.shap_values(input_array)

        # Normalize SHAP output to a 1D vector (n_features,) for sample 0 and chosen class
        # sv can be: list (per-class arrays), numpy array (2D or 3D), or a SHAP Explanation with .values
        if hasattr(sv, "values"):  # SHAP Explanation object (newer SHAP versions)
            vals = sv.values  # shape: (n_samples, n_features) or (n_samples, n_features, n_outputs)
            if vals.ndim == 2:
                shap_vec = vals[0]
            elif vals.ndim == 3:
                shap_vec = vals[0, :, class_idx]
            else:
                shap_vec = np.asarray(vals).reshape(-1)[:input_df.shape[1]]
        elif isinstance(sv, list):  # older multi-class: list of arrays [class][samples, features]
            shap_vec = np.asarray(sv[class_idx][0]).reshape(-1)
        elif isinstance(sv, np.ndarray):
            if sv.ndim == 2:
                shap_vec = sv[0]
            elif sv.ndim == 3:
                shap_vec = sv[0, :, class_idx]
            else:
                shap_vec = np.asarray(sv).reshape(-1)[:input_df.shape[1]]
        else:
            shap_vec = np.asarray(sv).reshape(-1)[:input_df.shape[1]]

        # Make sure values are 1D too
        values_vec = np.asarray(input_array[0]).reshape(-1)

        feature_names = list(input_df.columns)
        explanation = pd.DataFrame({
            'Feature': feature_names,
            'Value': values_vec,
            'SHAP_Impact': shap_vec,
            'Abs_Impact': np.abs(shap_vec)
        }).sort_values('Abs_Impact', ascending=False)

        print('SHAP Explanation:')
        for _, row in explanation.iterrows():
            impact = "↑" if row["SHAP_Impact"] > 0 else "↓"
            print(f'{row.Feature}: {row.Value} | {row.SHAP_Impact:+.4f} {impact}')

    except Exception as e:
        print(f'(⚠️ SHAP error for {model_name}: {e})')

# New function to load and train models (for UI)
def load_and_train_models(csv_path='Crop_recommendation-1.csv'):
    """Load data and train both models. Returns models and related objects."""
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1)
    y = df['label']
    
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
    
    # Train MLP
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
    
    # Create SHAP explainers
    shap_xgb = shap.TreeExplainer(xgb_model)
    shap_mlp = shap.KernelExplainer(mlp.predict_proba, X_train[:50])
    
    return {
        'xgb_model': xgb_model,
        'mlp': mlp,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist(),
        'xgb_acc': xgb_acc,
        'mlp_acc': mlp_acc,
        'shap_xgb': shap_xgb,
        'shap_mlp': shap_mlp
    }

# New function to get prediction with SHAP (for UI)
def get_prediction_with_shap(model, explainer, input_df, label_encoder):
    """Get prediction, confidence, and SHAP values."""
    input_array = np.ascontiguousarray(input_df.to_numpy(dtype=np.float64, copy=True))
    pred = model.predict(input_array)
    proba = model.predict_proba(input_array)[0]
    class_idx = int(np.argmax(proba))
    pred_crop = label_encoder.inverse_transform([pred[0]])[0]
    confidence = proba[class_idx] * 100
    
    # Get SHAP values
    sv = explainer.shap_values(input_array)
    
    if hasattr(sv, "values"):
        vals = sv.values
        if vals.ndim == 2:
            shap_vec = vals[0]
        elif vals.ndim == 3:
            shap_vec = vals[0, :, class_idx]
        else:
            shap_vec = np.asarray(vals).reshape(-1)[:input_df.shape[1]]
    elif isinstance(sv, list):
        shap_vec = np.asarray(sv[class_idx][0]).reshape(-1)
    elif isinstance(sv, np.ndarray):
        if sv.ndim == 2:
            shap_vec = sv[0]
        elif sv.ndim == 3:
            shap_vec = sv[0, :, class_idx]
        else:
            shap_vec = np.asarray(sv).reshape(-1)[:input_df.shape[1]]
    else:
        shap_vec = np.asarray(sv).reshape(-1)[:input_df.shape[1]]
    
    values_vec = np.asarray(input_array[0]).reshape(-1)
    
    return pred_crop, confidence, shap_vec, values_vec

def main():
    # 1. Load data
    df = pd.read_csv('Crop_recommendation-1.csv')
    X = df.drop('label', axis=1)
    y = df['label']
    # 2. Encode target
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    # 3. MLP needs scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)

    # --- XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    print('XGBoost accuracy:', accuracy_score(y_test, xgb_model.predict(X_test)))
    shap_xgb = shap.TreeExplainer(xgb_model)
    # --- Neural MLP
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    print('MLP accuracy:', accuracy_score(y_test, mlp.predict(X_test)))
    shap_mlp = shap.KernelExplainer(mlp.predict_proba, X_train[:50])

    # --- Get input data with option selection
    input_data = get_input_data(X)
    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=X.columns)
    
    # --- Make predictions with explanations
    explain_with_shap(xgb_model, shap_xgb, input_scaled, label_encoder, "XGBoost")
    explain_with_shap(mlp, shap_mlp, input_scaled, label_encoder, "MLP")

if __name__ == "__main__":
    main()
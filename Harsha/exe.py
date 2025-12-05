from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from harCode import load_and_train_models, fetch_live_weather, get_prediction_with_shap
import json

app = Flask(__name__)

# Load models once at startup
print("Loading models...")
models = load_and_train_models()
print("Models loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html', 
                         xgb_acc=f"{models['xgb_acc']*100:.2f}",
                         mlp_acc=f"{models['mlp_acc']*100:.2f}")

@app.route('/fetch_weather', methods=['POST'])
def get_weather():
    data = request.json
    location = data.get('location')
    
    weather = fetch_live_weather(location)
    
    if weather:
        return jsonify({
            'success': True,
            'temperature': weather['temperature'],
            'humidity': weather['humidity'],
            'rainfall': weather['rainfall']
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Could not fetch weather data'
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create input dataframe
        input_data = pd.DataFrame([[
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]], columns=models['feature_names'])
        
        # Scale input
        input_scaled = pd.DataFrame(
            models['scaler'].transform(input_data),
            columns=models['feature_names']
        )
        
        # Get predictions from both models
        xgb_crop, xgb_conf, xgb_shap, xgb_vals = get_prediction_with_shap(
            models['xgb_model'], models['shap_xgb'], 
            input_scaled, models['label_encoder']
        )
        
        mlp_crop, mlp_conf, mlp_shap, mlp_vals = get_prediction_with_shap(
            models['mlp'], models['shap_mlp'],
            input_scaled, models['label_encoder']
        )
        
        # Prepare SHAP data
        xgb_shap_data = [
            {
                'feature': models['feature_names'][i],
                'value': float(xgb_vals[i]),
                'shap': float(xgb_shap[i])
            }
            for i in range(len(models['feature_names']))
        ]
        xgb_shap_data.sort(key=lambda x: abs(x['shap']), reverse=True)
        
        mlp_shap_data = [
            {
                'feature': models['feature_names'][i],
                'value': float(mlp_vals[i]),
                'shap': float(mlp_shap[i])
            }
            for i in range(len(models['feature_names']))
        ]
        mlp_shap_data.sort(key=lambda x: abs(x['shap']), reverse=True)
        
        return jsonify({
            'success': True,
            'xgboost': {
                'crop': xgb_crop,
                'confidence': float(xgb_conf),
                'shap': xgb_shap_data
            },
            'mlp': {
                'crop': mlp_crop,
                'confidence': float(mlp_conf),
                'shap': mlp_shap_data
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
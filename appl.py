from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key'

# Global variables
model = None
scaler = None
label_encoder = None
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def load_and_train_model():
    global model, scaler, label_encoder
    
    try:
        print("Loading dataset...")
        df = pd.read_csv('Crop_recommendation.csv')
        print(f"Dataset loaded. Shape: {df.shape}")
        
        X = df[feature_names]
        y = df['label']
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"Model trained! Accuracy: {accuracy:.4f}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        
        # Validate and convert input
        input_data = [
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        
        # Scale input
        input_scaled = scaler.transform([input_data])
        
        # Get predictions
        probabilities = model.predict_proba(input_scaled)[0]
        crop_classes = label_encoder.classes_
        
        # Get top 3 recommendations
        top_indices = np.argsort(probabilities)[::-1][:3]
        recommendations = []
        
        for idx in top_indices:
            recommendations.append({
                'crop': crop_classes[idx],
                'confidence': round(float(probabilities[idx]) * 100, 2)
            })
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test')
def test():
    return jsonify({'message': 'Flask server is working!'})

if __name__ == '__main__':
    print("Starting server...")
    if load_and_train_model():
        print("Server starting on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model")
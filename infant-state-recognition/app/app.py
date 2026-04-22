import os
import joblib
import librosa
import numpy as np
from flask import Flask, request, render_template, jsonify
# Importing extraction tools directly or via module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import extract_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'outputs/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Models
MODEL_PATH = 'models/advanced_rf.pkl'
SCALER_PATH = 'models/scaler.pkl'
CLASS_MAP_PATH = 'models/class_mapping.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    class_mapping = joblib.load(CLASS_MAP_PATH)
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    print("Models loaded successfully.")
except Exception as e:
    model, scaler, inv_class_mapping = None, None, None
    print("Warning: Models not found. Run training scripts first.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained yet.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # Simple preprocessing (no augmentation for inference)
            audios, sr = preprocess_pipeline(filepath, augment=False)
            audio = audios[0] # Take the original unaugmented sound
            features = extract_features(audio, sr)
            
            # Predict
            X_scaled = scaler.transform([features])
            pred_idx = model.predict(X_scaled)[0]
            pred_class = inv_class_mapping[pred_idx]
            
            probs = model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probs)) * 100
            
            return jsonify({
                'prediction': pred_class,
                'confidence': f"{confidence:.2f}%",
                'status': 'success'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

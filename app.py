from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the model, scaler, selected features, and metadata
model_path = os.path.join(os.path.dirname(__file__), 'models', 'kidney_disease_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
features_path = os.path.join(os.path.dirname(__file__), 'models', 'selected_features.pkl')
metadata_path = os.path.join(os.path.dirname(__file__), 'models', 'model_metadata.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(features_path)
    
    # Load metadata if available
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        model_performance = metadata.get('model_performance', {})
    else:
        model_performance = {}
    
    print(f"Model loaded successfully!")
    print(f"Selected features: {selected_features}")
    # Debug: Print selected_features to check for non-numeric fields
    print("[DEBUG] selected_features:", selected_features)
    print(f"Model performance: {model_performance}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    selected_features = []

# Comprehensive feature definitions
feature_fullforms = {
    'age': 'Age',
    'bp': 'Blood Pressure (Systolic)',
    'sg': 'Specific Gravity (Urine)',
    'al': 'Albumin (Urine)',
    'su': 'Sugar (Urine)',
    'bgr': 'Blood Glucose Random',
    'bu': 'Blood Urea',
    'sc': 'Serum Creatinine',
    'sod': 'Sodium (Blood)',
    'pot': 'Potassium (Blood)',
    'hemo': 'Hemoglobin',
    'pcv': 'Packed Cell Volume',
    'wc': 'White Blood Cell Count',
    'rc': 'Red Blood Cell Count',
    'htn': 'Hypertension',
    'dm': 'Diabetes Mellitus',
    'cad': 'Coronary Artery Disease',
    'appet': 'Appetite',
    'pe': 'Pedal Edema',
    'ane': 'Anemia'
}

feature_ranges = {
    'age': '2 - 90 years',
    'bp': '50 - 180 mmHg',
    'sg': '1.005 - 1.025',
    'al': '0 - 5 (0=normal, 1-5=abnormal levels)',
    'su': '0 - 5 (0=normal, 1-5=abnormal levels)',
    'bgr': '70 - 490 mg/dL (Normal: 70-140)',
    'bu': '1.5 - 391 mg/dL (Normal: 7-20)',
    'sc': '0.4 - 15.0 mg/dL (Normal: 0.6-1.2)',
    'sod': '4.5 - 163 mEq/L (Normal: 135-145)',
    'pot': '2.5 - 47 mEq/L (Normal: 3.5-5.0)',
    'hemo': '3.1 - 17.8 g/dL (Normal: 12-16)',
    'pcv': '9 - 54% (Normal: 36-46%)',
    'wc': '2200 - 26400 cells/cumm (Normal: 4000-11000)',
    'rc': '2.1 - 8.0 millions/cmm (Normal: 4.5-5.5)',
    'htn': '0 = No, 1 = Yes',
    'dm': '0 = No, 1 = Yes',
    'cad': '0 = No, 1 = Yes',
    'appet': '0 = Poor, 1 = Good',
    'pe': '0 = No, 1 = Yes',
    'ane': '0 = No, 1 = Yes'
}

# Enhanced feature descriptions for better user understanding
feature_descriptions = {
    'age': 'Your current age in years',
    'bp': 'Your blood pressure reading (systolic/upper number)',
    'sg': 'Specific gravity of your urine sample',
    'al': 'Albumin protein level in urine (from lab test)',
    'su': 'Sugar/glucose level in urine (from lab test)',
    'bgr': 'Random blood glucose level (mg/dL)',
    'bu': 'Blood urea nitrogen level (mg/dL)',
    'sc': 'Serum creatinine level - key kidney function marker',
    'sod': 'Sodium level in blood (mEq/L)',
    'pot': 'Potassium level in blood (mEq/L)',
    'hemo': 'Hemoglobin level in blood (g/dL)',
    'pcv': 'Packed cell volume - percentage of red blood cells',
    'wc': 'White blood cell count (cells/cumm)',
    'rc': 'Red blood cell count (millions/cmm)',
    'htn': 'Do you have high blood pressure?',
    'dm': 'Do you have diabetes?',
    'cad': 'Do you have coronary artery disease?',
    'appet': 'How is your appetite?',
    'pe': 'Do you have swelling in feet/ankles?',
    'ane': 'Do you have anemia?'
}

def get_risk_level(probability):
    """Determine risk level based on probability with medical context"""
    if probability < 0.3:
        return "Low Risk", "#27ae60", "Your kidney function appears normal based on the provided parameters."
    elif probability < 0.7:
        return "Moderate Risk", "#f39c12", "Some indicators suggest potential kidney issues. Please consult a healthcare provider."
    else:
        return "High Risk", "#e74c3c", "Multiple indicators suggest kidney disease. Please seek immediate medical attention."

def validate_input_ranges(features):
    """Validate input values against expected medical ranges"""
    warnings = []
    
    # Define critical ranges for validation
    critical_ranges = {
        'age': (0, 120),
        'bp': (30, 250),
        'bgr': (50, 600),
        'bu': (0, 500),
        'sc': (0.1, 20.0),
        'hemo': (2.0, 20.0)
    }
    
    for feature, value in features.items():
        if feature in critical_ranges:
            min_val, max_val = critical_ranges[feature]
            if not (min_val <= value <= max_val):
                warnings.append(f"{feature_fullforms.get(feature, feature)}: {value} seems unusual")
    
    return warnings

@app.route('/')
def index():
    return render_template('index.html', 
                         features=selected_features, 
                         feature_fullforms=feature_fullforms, 
                         feature_ranges=feature_ranges,
                         feature_descriptions=feature_descriptions,
                         model_performance=model_performance)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500
    
    try:
        # Extract features from the form data
        features = {}
        for feature in selected_features:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({"error": f"Missing value for {feature_fullforms.get(feature, feature)}"}), 400
            features[feature] = float(value)
        
        # Validate input ranges
        warnings = validate_input_ranges(features)
        
        # Create a DataFrame with the features
        df = pd.DataFrame([features])
        
        # Ensure all selected features are present and in correct order
        df = df[selected_features]
        
        # Scale the features
        scaled_features = scaler.transform(df)
        
        # Predict using the loaded model
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        # Get risk level and interpretation
        risk_level, risk_color, interpretation = get_risk_level(probability)
        
        # Prepare response
        result = {
            "prediction": int(prediction),
            "probability": f"{probability * 100:.1f}%",
            "probability_decimal": round(probability, 3),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "interpretation": interpretation,
            "confidence": "High" if abs(probability - 0.5) > 0.3 else "Medium"
        }
        
        # Add warnings if any
        if warnings:
            result["warnings"] = warnings
        
        # Add model performance info
        if model_performance:
            result["model_info"] = {
                "accuracy": f"{model_performance.get('accuracy', 0) * 100:.1f}%",
                "reliability": "High" if model_performance.get('accuracy', 0) > 0.9 else "Good"
            }
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/model-info')
def model_info():
    """Endpoint to get model information"""
    if not model_performance:
        return jsonify({"error": "Model metadata not available"})
    
    return jsonify({
        "selected_features": selected_features,
        "performance": model_performance,
        "total_features": len(selected_features)
    })

if __name__ == "__main__":
    app.run(debug=True)
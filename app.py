import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Define paths for the model and scaler
model_path = 'logistic_regression_model.joblib'
scaler_path = 'scaler.joblib'

# Load the trained model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Define columns where 0 might represent a missing or invalid measurement
columns_to_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Mean values for these columns from the original training data
# These values are taken from the notebook's preprocessing step's mean_value variable before scaling.
# Specifically, from the state of X before scaling, X[col].mean() after replacing 0s with NaN.
means_for_zero_replacement = {
    'Glucose': 121.68676231976077,
    'BloodPressure': 72.40518414441017,
    'SkinThickness': 29.153419987873832,
    'Insulin': 155.54822283995818,
    'BMI': 32.457463672391015
}

# Feature names in the exact order as used for training
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded properly.'}), 500

    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert input data to a pandas DataFrame
        # Ensure the order of features matches the training data
        input_df = pd.DataFrame([data], columns=feature_names)

        # Replace 0 values with the pre-calculated means for specified columns
        for col in columns_to_replace_zero:
            if col in input_df.columns:
                input_df[col] = input_df[col].replace(0, means_for_zero_replacement[col])

        # Scale the input features using the loaded scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        # Prepare the response
        response = {
            'prediction': int(prediction),
            'probabilities': {
                'no_diabetes': float(prediction_proba[0]),
                'diabetes': float(prediction_proba[1])
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app on all available network interfaces
    # debug=False for production environment
    app.run(host='0.0.0.0', port=5000, debug=False)

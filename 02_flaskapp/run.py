from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle

app = Flask(__name__)

# Load the model components from the pickle file
with open('model_pipeline.pkl', 'rb') as f:
        kmeans, model, threshold = pickle.load(f)

def get_form_value(key, default=0.0, cast_type=float):
    try:
        return cast_type(request.form.get(key, default))
    except (TypeError, ValueError):
        return default

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Extract input data from the form with proper type casting
    input_data = [
        get_form_value('email_free', 0.0, float),
        get_form_value('home_phone_free', 0.0, float),
        get_form_value('mobile_phone_free', 0.0, float),
        get_form_value('other_cards', 0.0, float),
        get_form_value('location_check', 0.0, float),
        get_form_value('sign_out', 0.0, float),
        get_form_value('previous_address', 0.0, float),
        get_form_value('income', 0.0, float),
        get_form_value('name_email_similarity', 0.0, float),
        get_form_value('current_address_length', 0, int),
        get_form_value('age', 0, int),
        get_form_value('days_since_submission', 0.0, float),
        get_form_value('applications_same_zip', 0, int),
        get_form_value('applications_6_hours', 0, int),
        get_form_value('applications_24_hours', 0, int),
        get_form_value('applications_bank_branch', 0, int),
        get_form_value('emails_dob', 0, int),
        get_form_value('risk_score', 0.0, float),
        get_form_value('credit_limit', 0.0, float),
        get_form_value('session_length', 0, int),
        get_form_value('distinct_emails', 0.0, float),
        get_form_value('month', 0, int),
        get_form_value('payment_type', 0.0, float),
        get_form_value('employment_status', 0.0, float),
        get_form_value('housing_status', 0.0, float),
        get_form_value('internet', 0.0, float),
        get_form_value('device_os', 0.0, float)
    ]

    # Convert to numpy array and reshape for prediction
    input_data = np.array(input_data).reshape(1, -1)
        
    # Since there's no transformer, you can directly predict with the model
    fraud_proba = model.predict_proba(input_data)[0][1]
        
    result = 'Fraud' if fraud_proba > threshold else 'Not Fraud'

    return jsonify(result=result, probability=fraud_proba)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)

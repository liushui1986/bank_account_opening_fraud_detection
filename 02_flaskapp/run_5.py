from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle

app = Flask(__name__)

# Load the model components from the pickle file
with open('model_pipeline.pkl', 'rb') as f:
    kmeans, model, threshold = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract and convert data from the form
    form_data = [
        int(request.form.get('email', 0)),
        1 if request.form.get('home_phone') else 0,
        1 if request.form.get('mobile_phone') else 0,
        1 if request.form.get('other_cards') else 0,
        int(request.form.get('location_check', 0)),
        int(request.form.get('sign_out', 0)),
        1 if request.form.get('previous_address') else 0,
        request.form.get('income', type=float) or 0.0,
        request.form.get('name_email_similarity', type=float) or 0.0,
        int(request.form.get('living_length', 0)),
        int(request.form.get('age', 0)),
        request.form.get('days_since_submission', type=int) or 0,
        request.form.get('applications_zip', type=int) or 0,
        request.form.get('applications_6_hours', type=int) or 0,
        request.form.get('applications_24_hours', type=int) or 0,
        request.form.get('applications_bank_branch', type=int) or 0,
        request.form.get('emails_dob', type=int) or 0,
        request.form.get('risk_score', type=int) or 0,
        request.form.get('credit_limit', type=int) or 0,
        request.form.get('session_length', type=int) or 0,
        request.form.get('distinct_emails', type=int) or 0,
        request.form.get('month', type=int) or 0,
        request.form.get('payment_type', ''),
        request.form.get('employment_status', ''),
        request.form.get('housing_status', ''),
        int(request.form.get('internet', 0)),
        request.form.get('device_os', '')
    ]

    # Convert categorical variables to binary
    input_data = []
    input_data.extend(form_data[:21])
    input_data.extend([
        1 if request.form['payment_type'] == 'AA' else 0,
        1 if request.form['payment_type'] == 'AB' else 0,
        1 if request.form['payment_type'] == 'AC' else 0,
        1 if request.form['payment_type'] == 'AD' else 0,
        1 if request.form['payment_type'] == 'AE' else 0,
        1 if request.form['employment_status'] == 'CA' else 0,
        1 if request.form['employment_status'] == 'CB' else 0,
        1 if request.form['employment_status'] == 'CC' else 0,
        1 if request.form['employment_status'] == 'CD' else 0,
        1 if request.form['employment_status'] == 'CE' else 0,
        1 if request.form['employment_status'] == 'CF' else 0,
        1 if request.form['employment_status'] == 'CG' else 0,
        1 if request.form['housing_status'] == 'BA' else 0,
        1 if request.form['housing_status'] == 'BB' else 0,
        1 if request.form['housing_status'] == 'BC' else 0,
        1 if request.form['housing_status'] == 'BD' else 0,
        1 if request.form['housing_status'] == 'BE' else 0,
        1 if request.form['housing_status'] == 'BF' else 0,
        1 if request.form['housing_status'] == 'BG' else 0,
        1 if request.form['device_os'] == 'Linux' else 0,
        1 if request.form['device_os'] == 'Macintosh' else 0,
        1 if request.form['device_os'] == 'Other' else 0,
        1 if request.form['device_os'] == 'Windows' else 0,
        1 if request.form['device_os'] == 'X11' else 0,
    ])

    # Convert to numpy array and reshape for prediction
    input_data = np.array(input_data).reshape(1, -1)

    # Predict with the model
    prediction = model.predict(input_data)[0]
    fraud_proba = model.predict_proba(input_data)[0][1]

    result = 'Fraud' if fraud_proba > threshold else 'Not Fraud'

    return jsonify(result=result, probability=fraud_proba)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
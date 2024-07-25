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
    try:
        # Extracting and converting data from the form
        form_data = [
            int(request.form.get('email', 0)),
            request.form.get('home_phone', type=float, default=0.0),
            request.form.get('mobile_phone', type=float, default=0.0),
            request.form.get('other_cards', type=float, default=0.0),
            int(request.form.get('location_check', 0)),
            int(request.form.get('sign_out', 0)),
            request.form.get('previous_address', ''),
            request.form.get('income', type=float, default=0.0),
            request.form.get('name_email_similarity', type=float, default=0.0),
            int(request.form.get('living_length', 0)),
            int(request.form.get('age', 0)),
            request.form.get('days_since_submission', type=int, default=0),
            request.form.get('applications_zip', type=int, default=0),
            request.form.get('applications_6_hours', type=int, default=0),
            request.form.get('applications_24_hours', type=int, default=0),
            request.form.get('applications_bank_branch', type=int, default=0),
            request.form.get('emails_dob', type=int, default=0),
            request.form.get('risk_score', type=int, default=0),
            request.form.get('credit_limit', type=int, default=0),
            request.form.get('session_length', type=int, default=0),
            request.form.get('distinct_emails', type=int, default=0),
            request.form.get('month', type=int, default=0),
            request.form.get('payment_type', ''),
            request.form.get('employment_status', ''),
            request.form.get('housing_status', ''),
            int(request.form.get('internet', 0)),
            request.form.get('device_os', '')
        ]

        # Convert categorical variables to one-hot encoding
        payment_types = ['AA', 'AB', 'AC', 'AD', 'AE']
        employment_statuses = ['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG']
        housing_statuses = ['BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG']
        device_os_list = ['Linux', 'Macintosh', 'Other', 'Windows', 'X11']

        # One-hot encode categorical variables
        input_data = form_data[:21]
        input_data.extend([1 if request.form.get('payment_type') == p else 0 for p in payment_types])
        input_data.extend([1 if request.form.get('employment_status') == e else 0 for e in employment_statuses])
        input_data.extend([1 if request.form.get('housing_status') == h else 0 for h in housing_statuses])
        input_data.extend([1 if request.form.get('device_os') == d else 0 for d in device_os_list])

        # Convert to numpy array and reshape for prediction
        input_data = np.array(input_data).reshape(1, -1)

        # Predict with the model
        prediction = model.predict(input_data)[0]
        fraud_proba = model.predict_proba(input_data)[0][1]

        # Determine the result based on the threshold
        result = 'Fraud' if fraud_proba > threshold else 'Not Fraud'

        return jsonify(result=result, probability=fraud_proba)

    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
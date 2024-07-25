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
        form_data = {
            'email': request.form.get('email', ''),
            'home_phone': request.form.get('home_phone', ''),
            'mobile_phone': request.form.get('mobile_phone', ''),
            'other_cards': request.form.get('other_cards', ''),
            'location_check': request.form.get('location_check', ''),
            'sign_out': request.form.get('sign_out', ''),
            'previous_address': request.form.get('previous_address', ''),
            'income': request.form.get('income', ''),
            'name_email_similarity': request.form.get('name_email_similarity', ''),
            'living_length': request.form.get('living_length', ''),
            'age': request.form.get('age', ''),
            'days_since_submission': request.form.get('days_since_submission', ''),
            'applications_zip': request.form.get('applications_zip', ''),
            'applications_6_hours': request.form.get('applications_6_hours', ''),
            'applications_24_hours': request.form.get('applications_24_hours', ''),
            'applications_bank_branch': request.form.get('applications_bank_branch', ''),
            'emails_dob': request.form.get('emails_dob', ''),
            'risk_score': request.form.get('risk_score', ''),
            'credit_limit': request.form.get('credit_limit', ''),
            'session_length': request.form.get('session_length', ''),
            'distinct_emails': request.form.get('distinct_emails', ''),
            'month': request.form.get('month', ''),
            'payment_type': request.form.get('payment_type', ''),
            'employment_status': request.form.get('employment_status', ''),
            'housing_status': request.form.get('housing_status', ''),
            'internet': request.form.get('internet', ''),
            'device_os': request.form.get('device_os', '')
        }

        # Debug: Print raw form data
        print("Raw form data:", form_data)

        # Convert data to numeric types
        def convert(value, default=0.0):
            try:
                return float(value) if value else default
            except ValueError:
                return default

        # Prepare input data
        input_data = [
            convert(form_data['email'], 0),
            convert(form_data['home_phone']),
            convert(form_data['mobile_phone']),
            convert(form_data['other_cards']),
            convert(form_data['location_check'], 0),
            convert(form_data['sign_out'], 0),
            form_data['previous_address'],  # Assuming this is a string and not used in the model
            convert(form_data['income']),
            convert(form_data['name_email_similarity']),
            convert(form_data['living_length'], 0),
            convert(form_data['age'], 0),
            convert(form_data['days_since_submission'], 0),
            convert(form_data['applications_zip'], 0),
            convert(form_data['applications_6_hours'], 0),
            convert(form_data['applications_24_hours'], 0),
            convert(form_data['applications_bank_branch'], 0),
            convert(form_data['emails_dob'], 0),
            convert(form_data['risk_score'], 0),
            convert(form_data['credit_limit'], 0),
            convert(form_data['session_length'], 0),
            convert(form_data['distinct_emails'], 0),
            convert(form_data['month'], 0),
            form_data['payment_type'],
            form_data['employment_status'],
            form_data['housing_status'],
            convert(form_data['internet'], 0),
            form_data['device_os']
        ]

        # Convert categorical variables to one-hot encoding
        payment_types = ['AA', 'AB', 'AC', 'AD', 'AE']
        employment_statuses = ['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG']
        housing_statuses = ['BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG']
        device_os_list = ['Linux', 'Macintosh', 'Other', 'Windows', 'X11']

        # One-hot encoding
        input_data.extend([1 if form_data['payment_type'] == p else 0 for p in payment_types])
        input_data.extend([1 if form_data['employment_status'] == e else 0 for e in employment_statuses])
        input_data.extend([1 if form_data['housing_status'] == h else 0 for h in housing_statuses])
        input_data.extend([1 if form_data['device_os'] == d else 0 for d in device_os_list])

        # Debug: Print processed input data
        print("Processed input data:", input_data)

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
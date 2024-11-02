# app/routes.py

from flask import Blueprint, render_template, request, jsonify
import pickle
import numpy as np

# Define a blueprint
main = Blueprint('main', __name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data using `request.form` since data is sent in URL-encoded format
    time = float(request.form.get('time'))
    amount = float(request.form.get('amount'))

    # Predict using the model
    prediction = model.predict([[time, amount]])[0]
    result = "Fraudulent" if prediction == 1 else "Legit"

    # Return JSON result
    return jsonify({'result': result})
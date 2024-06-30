# Define a single /predict endpoint that takes in a JSON payload, uses the loaded model to make a prediction, and returns the prediction as a JSON response.
import json
import os
from flask import Flask, request, jsonify
import pandas as pd

from src.preprocess import Preprocessor

MODEL_PATH = os.getenv("MODEL_PATH", default="models/titanic_random_forest_model.pkl")

# Load the model
import joblib
loaded_model = joblib.load(MODEL_PATH)

app = Flask(__name__)

# Define a home route that returns a simple json response
@app.route("/")
def home():
    return jsonify({"message": "Hello, Titanic!"})

@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON payload from the request
    payload = request.get_json()

    # Convert the JSON payload to a DataFrame
    data = pd.DataFrame(payload)

    # Preprocess the data
    preprocessor = Preprocessor()
    data_df = preprocessor.preprocess(data)

    # Use the loaded model to make a prediction
    prediction = loaded_model.predict(data_df)

    # Return the prediction as a JSON response
    return jsonify(prediction.tolist())

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000, debug=True)

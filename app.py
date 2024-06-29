# Define a single /predict endpoint that takes in a JSON payload, uses the loaded model to make a prediction, and returns the prediction as a JSON response.
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON payload from the request
    data = request.get_json()
    # Convert the JSON payload to a DataFrame
    data_df = pd.DataFrame(data)
    # Use the loaded model to make a prediction
    prediction = loaded_model.predict(data_df)
    # Return the prediction as a JSON response
    return jsonify(prediction.tolist())

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000)

# Define a single /predict endpoint that takes in a JSON payload, uses the loaded model to make a prediction, and returns the prediction as a JSON response using FastAPI.

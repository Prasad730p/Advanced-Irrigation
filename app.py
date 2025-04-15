from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model and scaler
MODEL_PATH = "pump_time_model.pkl"
SCALER_PATH = "scaler.pkl"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract input values
        features = [
            data.get("moisture"),
            data.get("temperature"),
            data.get("humidity"),
            data.get("evaporation"),
            data.get("rain")
        ]
        
        # Validate input
        if None in features:
            return jsonify({"error": "Missing input parameters"}), 400
        
        # Convert input to NumPy array and scale
        input_data = np.array([features])
        input_data_scaled = scaler.transform(input_data)
        
        # Predict using the model
        prediction = model.predict(input_data_scaled)[0]
        
        return jsonify({"pump_on_time": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
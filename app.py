import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Path to your pickled decision tree regression model
MODEL_PATH = "decision_regression_pkl.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure it is placed in the root directory.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Load the model globally on application startup
model = load_model()

# Dynamically extract feature inputs expected by the training state
# Standard features: ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission']
FEATURES = list(model.feature_names_in_)

@app.route("/", methods=["GET"])
def home():
    # Renders your frontend UI and passes feature names for building form inputs dynamically
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Support both multi-part Form submissions and raw API JSON payloads
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Parse inputs into an array maintaining the correct feature column sequence
        input_data = []
        for feature in FEATURES:
            if feature not in data or data[feature] == "":
                return jsonify({"error": f"Missing expected parameter: {feature}"}), 400
            
            val = data[feature]
            # Convert numeric entries while leaving encoded category strings intact
            try:
                if "." in str(val):
                    input_data.append(float(val))
                else:
                    input_data.append(int(val))
            except ValueError:
                input_data.append(val)
        
        # Build DataFrame with proper feature names to satisfy Scikit-Learn requirements
        input_df = pd.DataFrame([input_data], columns=FEATURES)
        
        # Perform decision tree regression prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "success": True,
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render routes web traffic via an automatically assigned PORT variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

MODEL_PATH = "decision_regression_pkl.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in the root directory.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Global model initialization
try:
    model = load_model()
    # Extract exact features: ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission']
    FEATURES = list(model.feature_names_in_)
except Exception as e:
    print(f"Error initializing model: {e}")
    FEATURES = ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission']

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept both form data and JSON payloads seamlessly
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Build dictionary aligning with expected features
        parsed_data = {}
        for feature in FEATURES:
            if feature not in data or str(data[feature]).strip() == "":
                return jsonify({"error": f"Missing or empty input for field: {feature}"}), 400
            
            raw_val = str(data[feature]).strip()
            
            # Strict type handling to prevent Scikit-Learn data type mismatch errors
            if feature in ["Year", "Mileage"]:
                try:
                    parsed_data[feature] = [int(float(raw_val))]
                except ValueError:
                    return jsonify({"error": f"Field '{feature}' must be a valid integer numeric value."}), 400
            elif feature in ["Engine Size"]:
                try:
                    parsed_data[feature] = [float(raw_val)]
                except ValueError:
                    return jsonify({"error": f"Field '{feature}' must be a valid decimal numeric value."}), 400
            else:
                # Keep text categories as strings. 
                # Note: If your model throws a "could not convert string to float" error here, 
                # you must pass the exact encoded numeric representation (ID/Integer) used during training.
                parsed_data[feature] = [raw_val]

        # Convert to DataFrame ensuring column sequence matches model expectations perfectly
        input_df = pd.DataFrame(parsed_data)[FEATURES]
        
        # Generate model prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "success": True,
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        # Detailed error printing inside Render logs to debug exact mismatches
        print(f"Prediction Pipeline Exception: {str(e)}")
        return jsonify({"error": f"Server Prediction Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


        
    

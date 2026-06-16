import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

MODEL_PATH = "decision_regression_pkl.pkl"

# Safety Fallback List of Features
FEATURES = ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission']

def load_model_safely():
    global FEATURES
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: {MODEL_PATH} not found in root directory!")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded_model = pickle.load(f)
            
        # Try to dynamically read features stored inside the pickle file
        if hasattr(loaded_model, "feature_names_in_"):
            FEATURES = list(loaded_model.feature_names_in_)
            print(f"SUCCESS: Loaded model features directly: {FEATURES}")
        return loaded_model
    except Exception as e:
        print(f"ERROR DURING INITIAL MODEL EXTRACTION: {str(e)}")
        return None

# Load the model into global state safely
model = load_model_safely()

@app.route("/", methods=["GET"])
def home():
    # If HTML is not found, this will return an absolute error string instead of breaking Render
    try:
        return render_template("index.html", features=FEATURES)
    except Exception as e:
        return f"<h3>Flask Application is Online, but index.html is missing inside your templates/ folder. Error: {str(e)}</h3>", 500

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        # Reload attempt just in case worker lost context
        model = load_model_safely()
        if model is None:
            return jsonify({"error": "The ML model file could not be read or initialized by the web worker."}), 500

    try:
        # Gracefully handle normal text forms or JSON requests
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Build clean ordered dataframe inputs matching the required 7 features
        parsed_data = {}
        for feature in FEATURES:
            if feature not in data or str(data[feature]).strip() == "":
                return jsonify({"error": f"Value for '{feature}' is missing or empty."}), 400
            
            raw_value = str(data[feature]).strip()
            
            # Numeric Feature Conversions
            if feature in ["Year", "Mileage"]:
                parsed_data[feature] = [int(float(raw_value))]
            elif feature in ["Engine Size"]:
                parsed_data[feature] = [float(raw_value)]
            else:
                # Text variables passed down natively
                parsed_data[feature] = [raw_value]

        # Order column sequence exactly as array expects
        input_df = pd.DataFrame(parsed_data)[FEATURES]
        
        # Calculate prediction via Decision Tree
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "success": True,
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        print(f"CRITICAL CRASH ON PREDICT ROUTE: {str(e)}")
        return jsonify({"error": f"Internal Application Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

         


        
    

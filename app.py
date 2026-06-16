import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

MODEL_PATH = "decision_regression_pkl.pkl"

# Hardcoded features matching your model's exact signature to guarantee no initialization failures
FEATURES = ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission']

def load_model_safely():
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: {MODEL_PATH} not found in root directory!")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"ERROR LOADING PICKLE FILE: {str(e)}")
        return None

# Load the model globally on bootup
model = load_model_safely()

# Pure HTML/CSS Embedded template with attractive glassmorphic design and smooth AJAX handling
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Regression Model Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: radial-gradient(circle at 10% 20%, rgb(90, 92, 234) 0%, rgb(32, 33, 71) 90%);
        }
        .glass-panel {
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4 sm:p-6 md:p-8 text-white">

    <div class="w-full max-w-2xl glass-panel rounded-3xl p-6 md:p-10 shadow-2xl relative overflow-hidden transition-all duration-300">
        <div class="absolute -top-24 -right-24 w-48 h-48 bg-indigo-500 rounded-full blur-3xl opacity-40"></div>
        
        <header class="mb-8 relative z-10">
            <h1 class="text-3xl font-bold tracking-tight bg-gradient-to-r from-white via-indigo-200 to-indigo-400 bg-clip-text text-transparent">
                ML Regressor Model Estimation
            </h1>
            <p class="text-indigo-200/70 text-sm mt-1">Provide the parameters below to compute your model evaluation target.</p>
        </header>

        <form id="predictionForm" class="space-y-6 relative z-10">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-5">
                {% for feature in features %}
                <div class="flex flex-col space-y-2">
                    <label for="{{ feature }}" class="text-xs font-semibold uppercase tracking-wider text-indigo-200">
                        {{ feature }}
                    </label>
                    <input 
                        type="text" 
                        name="{{ feature }}" 
                        id="{{ feature }}" 
                        required
                        placeholder="Enter {{ feature }} value..."
                        class="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/20 transition-all text-sm"
                    >
                </div>
                {% endfor %}
            </div>

            <button 
                type="submit" 
                class="w-full bg-gradient-to-r from-indigo-500 to-violet-600 hover:from-indigo-600 hover:to-violet-700 text-white font-semibold py-3.5 px-6 rounded-xl transition-all shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/40 cursor-pointer active:scale-[0.99] mt-4"
            >
                Generate Prediction
            </button>
        </form>

        <div id="resultContainer" class="hidden mt-8 pt-6 border-t border-white/10 relative z-10 transition-all duration-500">
            <div class="p-5 rounded-2xl bg-indigo-950/40 border border-indigo-500/20 flex flex-col items-center justify-center text-center">
                <span class="text-indigo-300 font-medium text-xs uppercase tracking-widest mb-1">Predicted Value Output</span>
                <div id="predictionOutput" class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-green-300 to-emerald-400">
                    --
                </div>
            </div>
        </div>

        <div id="errorContainer" class="hidden mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-sm text-center relative z-10"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const form = e.target;
            const formData = new FormData(form);
            const resultContainer = document.getElementById('resultContainer');
            const predictionOutput = document.getElementById('predictionOutput');
            const errorContainer = document.getElementById('errorContainer');

            errorContainer.classList.add('hidden');
            resultContainer.classList.add('hidden');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok && result.success) {
                    predictionOutput.innerText = result.prediction.toLocaleString();
                    resultContainer.classList.remove('hidden');
                    resultContainer.scrollIntoView({ behavior: 'smooth' });
                } else {
                    throw new Error(result.error || 'Failed to parse predictive values.');
                }
            } catch (err) {
                errorContainer.innerText = err.message;
                errorContainer.classList.remove('hidden');
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    # Dynamic runtime string rendering prevents template directory lookup entirely
    return render_template_string(HTML_LAYOUT, features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = load_model_safely()
        if model is None:
            return jsonify({"error": "The decision_regression_pkl.pkl file could not be read or loaded."}), 500

    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        parsed_data = {}
        for feature in FEATURES:
            if feature not in data or str(data[feature]).strip() == "":
                return jsonify({"error": f"Missing or empty input for field: {feature}"}), 400
            
            raw_value = str(data[feature]).strip()
            
            # Numeric Feature Conversions
            if feature in ["Year", "Mileage"]:
                parsed_data[feature] = [int(float(raw_value))]
            elif feature in ["Engine Size"]:
                parsed_data[feature] = [float(raw_value)]
            else:
                # String conversion fallback if categories are strings (e.g. "Toyota", "Diesel")
                # If your model was trained on encoded numbers (e.g. 0, 1, 2), type those integers into the form fields.
                try:
                    if raw_value.isdigit():
                        parsed_data[feature] = [int(raw_value)]
                    else:
                        parsed_data[feature] = [float(raw_value)]
                except ValueError:
                    parsed_data[feature] = [raw_value]

        # Convert into Ordered DataFrame structure matching columns
        input_df = pd.DataFrame(parsed_data)[FEATURES]
        
        # Calculate tree prediction matrix output
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "success": True,
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        print(f"PREDICTION CRASH ERROR: {str(e)}")
        return jsonify({"error": f"Model error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
      
   
            
       

         


        
    

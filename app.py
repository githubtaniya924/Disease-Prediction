from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained models and symptom list once when the app starts
try:
    dt_model = joblib.load('models/dt_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    association_rules = joblib.load('models/association_rules.pkl') # Load the association rules here
    symptom_columns = joblib.load('models/symptom_columns.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Please ensure they are in the 'models' directory.")
    exit()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/symptoms")
def get_symptoms():
    return jsonify({'all_symptoms': symptom_columns})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    selected_symptoms = data.get('symptoms', [])
    model_choice = data.get('model', 'decision_tree')

    # Create a one-hot encoded DataFrame for the prediction input
    input_df = pd.DataFrame(np.zeros((1, len(symptom_columns))), columns=symptom_columns)
    
    for symptom in selected_symptoms:
        if symptom in input_df.columns:
            input_df[symptom] = 1

    predicted_disease = "No prediction available"
    confidence = "N/A"
    
    # Store the results for the Association Rules model
    related_diseases = []

    if model_choice == 'decision_tree':
        predicted_disease = dt_model.predict(input_df)[0]
        confidence = np.max(dt_model.predict_proba(input_df))
    elif model_choice == 'knn':
        predicted_disease = knn_model.predict(input_df)[0]
        confidence = np.max(knn_model.predict_proba(input_df))
    elif model_choice == 'association':
        # Find rules where the user's selected symptoms are a subset of the antecedents
        # The loaded file is a list of dictionaries, not a DataFrame, so we iterate directly.
        for rule in association_rules:
            antecedents = rule['antecedents']
            consequents = rule['consequents']
            
            # Check if all selected symptoms are in the rule's antecedents
            if all(symptom in antecedents for symptom in selected_symptoms):
                # Add the diseases from the matching rule to the list
                related_diseases.extend(consequents)
        
        # Remove duplicates and format the result
        if related_diseases:
            related_diseases = list(set(related_diseases))
            return jsonify({
                'rules': ", ".join(related_diseases)
            })
        else:
            return jsonify({
                'rules': "No relevant rules found for the selected symptoms."
            })
    
    # Return the prediction and confidence for DT and KNN
    return jsonify({
        'disease': predicted_disease,
        'confidence': f"{confidence:.2%}" if isinstance(confidence, float) else confidence
    })

if __name__ == "__main__":
    app.run(debug=True)

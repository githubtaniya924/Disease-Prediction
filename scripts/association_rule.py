# association_rules_disease_fixed_v2.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import os

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv('Disease_Symptom_Dataset.csv')  # First column must be 'disease'

# ----------------------------
# Sample and shuffle dataset
# ----------------------------
if len(df) > 70000:
    df = df.sample(n=70000, random_state=42)
else:
    df = df.sample(frac=1, random_state=42)  # shuffle full dataset if less than 50k

df = df.reset_index(drop=True)  # reset index after shuffling

# ----------------------------
# Prepare symptom columns only
# ----------------------------
symptom_columns = df.columns[1:]  # assume first column is 'disease'
symptom_df = df[symptom_columns].astype(bool)  # convert to boolean

# ----------------------------
# Generate frequent itemsets
# ----------------------------
frequent_itemsets = apriori(symptom_df, min_support=0.005, use_colnames=True)  # low support for rare combos

# ----------------------------
# Generate association rules
# ----------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# ----------------------------
# Map antecedents to actual diseases
# ----------------------------
assoc_rules_list = []
for _, row in rules.iterrows():
    antecedent_symptoms = list(row['antecedents'])

    # Find all diseases where these symptoms exist
    matched_diseases = df[df[antecedent_symptoms].all(axis=1)]['diseases'].unique().tolist()

    if matched_diseases:  # keep only rules with at least one disease
        assoc_rules_list.append({
            'antecedents': antecedent_symptoms,
            'consequents': matched_diseases,
            'confidence': row['confidence']
        })

# ----------------------------
# Save rules
# ----------------------------
os.makedirs('models', exist_ok=True)
joblib.dump(assoc_rules_list, 'association_rules.pkl')
print(f"Association rules for disease prediction saved! Total rules: {len(assoc_rules_list)}")
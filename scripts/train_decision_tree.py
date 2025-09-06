import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)
import os

# --- Create folders ---
os.makedirs('../models', exist_ok=True)
os.makedirs('../static', exist_ok=True)

# --- Load dataset ---
df = pd.read_csv('C:/DWDM-Project/Disease-Prediction/Disease_Symptom_Dataset.csv')# use relative path

# --- Data preprocessing ---
df.iloc[:, 1:] = df.iloc[:, 1:].fillna(0)
X = df.drop('diseases', axis=1)
y = df['diseases']

# --- Filter rare diseases ---
rare_diseases = y.value_counts()[y.value_counts() < 2].index
X_filtered = X[~y.isin(rare_diseases)]
y_filtered = y[~y.isin(rare_diseases)]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)

# --- Train Decision Tree ---
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = dt_model.predict(X_test)
print("--- DecisionTree Metrics ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))

# --- Save model and graphs ---
joblib.dump(dt_model, '../models/DecisionTree_model.pkl')
joblib.dump(X_filtered.columns.tolist(), '../models/symptom_columns.pkl')


# Accuracy per class
class_acc = {}
for cls in dt_model.classes_:
    idx = y_test == cls
    class_acc[cls] = accuracy_score(y_test[idx], y_pred[idx])

# Sort and take top 10
top_classes = dict(sorted(class_acc.items(), key=lambda item: item[1], reverse=True)[:10])

plt.figure(figsize=(10,6))
plt.bar(top_classes.keys(), top_classes.values(), color='skyblue')
plt.xticks(rotation=90)
plt.ylabel('Accuracy')
plt.title('Top 10 Disease Class Accuracies')
plt.tight_layout()
plt.savefig('../static/DecisionTree_top10_accuracy.png')
plt.close()

#F1 Score & Recall per Class
top_classes = y_test.value_counts().index[:50]

# Compute per-class F1-score and Recall
from sklearn.metrics import f1_score, recall_score

f1_scores = f1_score(y_test, y_pred, labels=top_classes, average=None)
recall_scores = recall_score(y_test, y_pred, labels=top_classes, average=None)

# Create a DataFrame for plotting
import pandas as pd
df_scores = pd.DataFrame({
    'Class': top_classes,
    'F1_score': f1_scores,
    'Recall': recall_scores
})
plt.figure(figsize=(16,6))
plt.plot(df_scores['Class'], df_scores['F1_score'], marker='o', linestyle='-', color='green', label='F1 Score')
plt.plot(df_scores['Class'], df_scores['Recall'], marker='x', linestyle='--', color='orange', label='Recall')
plt.xticks(rotation=90)
plt.ylabel('Score')
plt.title('F1 Score & Recall per Class')
plt.legend()
plt.tight_layout()
plt.savefig('../static/DecisionTree_F1_Recall_line.png')
plt.close()


print("DecisionTree model and graphs saved successfully!")

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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
df = pd.read_csv('C:/DWDM-Project/Disease-Prediction/Disease_Symptom_Dataset.csv')  # use relative path

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

# --- Train KNN ---
k_model = KNeighborsClassifier(n_neighbors=5)
k_model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = k_model.predict(X_test)
print("--- KNN Metrics ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))

# --- Save model and graphs ---
joblib.dump(k_model, '../models/KNN_model.pkl')
joblib.dump(X_filtered.columns.tolist(), '../models/symptom_columns.pkl')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=k_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=k_model.classes_)
disp.plot(xticks_rotation=90)
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig('../static/KNN_confusion_matrix.png')
plt.close()

# Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(k_model, X_test, y_test, name="KNN")
plt.title("KNN Precision-Recall Curve")
plt.tight_layout()
plt.savefig('../static/KNN_pr_curve.png')
plt.close()

print("KNN model and graphs saved successfully!")

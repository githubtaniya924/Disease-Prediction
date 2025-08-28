import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_dataframe(df):
    # Handle Missing Values
    df = df.replace(np.nan, 'None')

    # Handle Inconsistent Formatting (Whitespace and Case)
    # The to_lower() method is more robust than a loop
    df.columns = df.columns.str.strip().str.lower()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.lower()

    # Handle Duplicate Records
    df.drop_duplicates(inplace=True)

    return df

# Load and clean all datasets
df_main = clean_dataframe(pd.read_csv('dataset.csv'))
df_severity = clean_dataframe(pd.read_csv('Symptom-severity.csv'))
df_description = clean_dataframe(pd.read_csv('symptom_Description.csv'))
df_precaution = clean_dataframe(pd.read_csv('symptom_precaution.csv'))

# Step 1: Merging the Descriptive Data
df_disease_info = pd.merge(df_description, df_precaution, on='disease')

print("Merged Disease Info DataFrame shape:", df_disease_info.shape)
print("\nFirst 5 rows of merged info:")
print(df_disease_info.head())

#Dataset Integration: Datasets- dataset.csv and Symptom-severity.csv
symptom_weights = df_severity.set_index('symptom')['weight'].to_dict()
print("Symptom weights dictionary created.")

# Use the lowercase column name 'disease'
symptom_cols = [col for col in df_main.columns if col != 'disease']
print("List of symptom columns:")
print(symptom_cols)

# Convert the symptom names to their weights using a single replacement operation
# The 'replace' method is much more efficient than looping
df_weighted = df_main.replace(symptom_weights)
print(df_weighted.head())

# The 'none' values and blanks might need to be converted to 0
# This is a critical step to ensure all symptom columns are numerical
df_weighted[symptom_cols] = df_weighted[symptom_cols].replace({'none': 0, ' ' : 0}).fillna(0)

# The final output should have numerical values in symptom columns
print("Weighted DataFrame shape:", df_weighted.shape)
print("\nFirst 5 rows of weighted data:")
print(df_weighted.head())



#Separate Features and Target
# Create a list of symptom columns to serve as your features
X_cols = [col for col in df_weighted.columns if col != 'disease']

# Assign the features (symptom data) to X and the target (disease names) to y
X = df_weighted[X_cols]
y = df_weighted['disease']

print("Shape of X (Features):", X.shape)
print("Shape of y (Target):", y.shape)


#Encode the Target Variable
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()

# Fit the encoder to your target variable and transform it
y = le.fit_transform(y)

print("Encoded target variable (y):", y)
print("Mapping of original names to encoded labels:")
# You can get the mapping to understand what each number represents
for i, name in enumerate(le.classes_):
    print(f"{i}: {name}")


#Split the Data into Training and Testing Sets : Divide the dataset into 2 subsets- trainig(80%) and Testing(20%)
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
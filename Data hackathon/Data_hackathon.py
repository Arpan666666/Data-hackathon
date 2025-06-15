# land_cover_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv(r"C:\Users\ARPAN MANDAL\Downloads\processed_ndvi.csv")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv(r"C:\Users\ARPAN MANDAL\Downloads\processed_ndvi.csv")  # <-- change filename if needed

# Strip extra whitespace from column names
data.columns = data.columns.str.strip()

# Print all column names to verify
print("Column names:", data.columns.tolist())

# Identify ID and target columns based on printed names
# Replace these with exact names if needed
id_columns = ['V1', 'V2']         # ID columns (drop these)
target_column = 'V3'              # Target column (e.g. 'V3' or '∆ V3')

# Drop ID columns and separate features/labels
X = data.drop(columns=id_columns + [target_column])
y = data[target_column]

# Encode target labels (e.g. water, forest → 0, 1, ...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Example: Predicting for new data (replace with real input)
# Sample input must match number and order of feature columns
# Example: random values (replace with actual row if needed)
sample_input = [X.iloc[0].values]  # Using the first row for example

predicted_class_index = clf.predict(sample_input)[0]
predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]

print("\nPredicted land cover class for sample input:", predicted_class_name)
# <-- change filename if needed

# Strip extra whitespace from column names
data.columns = data.columns.str.strip()

# Print all column names to verify
print("Column names:", data.columns.tolist())

# Identify ID and target columns based on printed names
# Replace these with exact names if needed
id_columns = ['V1', 'V2']         # ID columns (drop these)
target_column = 'V3'              # Target column (e.g. 'V3' or '∆ V3')

# Drop ID columns and separate features/labels
X = data.drop(columns=id_columns + [target_column])
y = data[target_column]

# Encode target labels (e.g. water, forest → 0, 1, ...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Example: Predicting for new data (replace with real input)
# Sample input must match number and order of feature columns
# Example: random values (replace with actual row if needed)
sample_input = [X.iloc[0].values]  # Using the first row for example

predicted_class_index = clf.predict(sample_input)[0]
predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]

print("\nPredicted land cover class for sample input:", predicted_class_name)

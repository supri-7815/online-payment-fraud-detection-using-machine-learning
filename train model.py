import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
# Replace with your dataset file name
data = pd.read_csv("payments.csv")

# Display first rows
print("Dataset Loaded Successfully")
print(data.head())

# Encode transaction type (if it is categorical)
if data['type'].dtype == 'object':
    le = LabelEncoder()
    data['type'] = le.fit_transform(data['type'])

# Select features and target
X = data[['step', 'type', 'amount', 'oldbalanceOrg',
          'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]

y = data['isFraud']   # Target column

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("payments.pkl", "wb"))
print("Model saved as payments.pkl")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

# Define paths
base_dir = "C:/Users/muthu/Downloads/crop/"
csv_path = os.path.join(base_dir, "Crop_recommendation.csv")
model_path = os.path.join(base_dir, "crop_model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")
le_path = os.path.join(base_dir, "label_encoder.pkl")
feature_names_path = os.path.join(base_dir, "feature_names.pkl")

# Verify CSV file exists
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}. Please ensure Crop_recommendation.csv is in {base_dir}.")
    exit(1)

# Load and preprocess data
try:
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully. Shape:", df.shape)
except Exception as e:
    print(f"Error loading CSV: {str(e)}")
    exit(1)

# Data preprocessing
df = df.dropna()
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[feature_names]
y = df['label_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save artifacts
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(le, le_path)
joblib.dump(feature_names, feature_names_path)
print(f"Files saved: {model_path}, {scaler_path}, {le_path}, {feature_names_path}")

# Generate and save visualizations
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))
plt.close()

feature_importance = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.savefig(os.path.join(base_dir, 'feature_importance.png'))
plt.close()

# Test prediction with DataFrame
test_input = np.array([[90, 42, 43, 20.8797, 82.0027, 6.5030, 202.9355]])
test_df = pd.DataFrame(test_input, columns=feature_names)
test_scaled = scaler.transform(test_df)
test_pred = model.predict(test_scaled)
print(f"Test Prediction: {le.inverse_transform(test_pred)[0]}")
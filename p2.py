import joblib
import numpy as np
import pandas as pd
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
base_dir = "C:/Users/muthu/Downloads/crop/"
model_path = os.path.join(base_dir, "crop_model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")
le_path = os.path.join(base_dir, "label_encoder.pkl")
feature_names_path = os.path.join(base_dir, "feature_names.pkl")

# Load the model, scaler, label encoder, and feature names
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    feature_names = joblib.load(feature_names_path)
    logging.info("Successfully loaded model, scaler, label encoder, and feature names.")
except FileNotFoundError as e:
    logging.error(f"Error: One or more files not found in {base_dir}. Run p1.py first. Error: {str(e)}")
    exit(1)
except Exception as e:
    logging.error(f"Error loading files: {str(e)}")
    exit(1)

# Prepare test input
test_input = np.array([[90, 42, 43, 20.8797, 82.0027, 6.5030, 202.9355]])
try:
    test_df = pd.DataFrame(test_input, columns=feature_names)
    logging.info(f"Test DataFrame created with shape: {test_df.shape}")
    scaled_data = scaler.transform(test_df)
    logging.info(f"Scaled data shape: {scaled_data.shape}")
    pred = model.predict(scaled_data)
    prediction = le.inverse_transform(pred)[0]
    logging.info(f"Prediction: {prediction}")
    print(f"Prediction: {prediction}")
except Exception as e:
    logging.error(f"Prediction failed: {str(e)}")
    exit(1)
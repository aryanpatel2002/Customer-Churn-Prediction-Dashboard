import pandas as pd
import joblib

# Load model + columns
def load_model():
    model = joblib.load("models/churn_model.pkl")          # trained model
    model_columns = joblib.load("models/model_columns.pkl")  # feature names used in training
    return model, model_columns


# Preprocess Data
def preprocess_data(df: pd.DataFrame, model_columns: list):
    """
    Convert raw input dataframe into the same format as training data
    - Handles encoding
    - Ensures correct column order
    - Adds missing columns with 0
    """

    df = df.copy()

    # Example: map categorical to numeric (adjust based on your dataset)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    if "HasCrCard" in df.columns:
        df["HasCrCard"] = df["HasCrCard"].astype(int)

    if "IsActiveMember" in df.columns:
        df["IsActiveMember"] = df["IsActiveMember"].astype(int)

    # Handle one-hot encoding for categorical features
    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    return df

# SHAP Explainer
import shap

def shap_explainer(model, sample_data):
    """Return SHAP values for given model and data"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_data)
    return shap_values
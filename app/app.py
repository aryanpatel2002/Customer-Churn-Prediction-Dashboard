import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap

from utils import load_model, preprocess_data

# Load model
model, model_columns = load_model()

# Load CSS
try:
    with open("customer-churn-dashboard/app/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("`style.css` not found. The app will use default styling.")

# -----------------------------
# App Header
# -----------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide", page_icon="📊")
st.title("📊 Customer Churn Predictor Dashboard")
st.markdown("Predict customer churn, explore insights, and visualize feature importance.")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["🏠 Home", "👤 Single Prediction", "📂 Batch Prediction", "📈 Model Insights"])

# -----------------------------
# Pages
# -----------------------------
if option == "🏠 Home":
    st.markdown(
        """
        Welcome to the **Customer Churn Predictor Dashboard**!  
        Use the sidebar to:
        - Predict churn for a single customer  
        - Upload a dataset for batch predictions  
        - Explore model insights and feature importance
        """
    )

elif option == "👤 Single Prediction":
    st.subheader("Predict churn for a single customer")

    with st.form(key="single_form"):
        col1, col2 = st.columns(2)

        with col1:
            credit_score = st.number_input("Credit Score", 300, 900, 600)
            age = st.number_input("Age", 18, 100, 35)
            tenure = st.slider("Tenure (Years)", 0, 10, 3)
            balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)

        with col2:
            estimated_salary = st.number_input("Estimated Salary", 0, 200000, 50000)
            gender = st.selectbox("Gender", ["Male", "Female"])
            has_card = st.selectbox("Has Credit Card?", [0, 1])
            is_active = st.selectbox("Active Member?", [0, 1])

        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
    # Convert Gender to 0/1
        gender_val = 1 if gender == "Male" else 0

    # Create raw input DataFrame
        input_df = pd.DataFrame([[credit_score, gender_val, age, tenure, balance, has_card, is_active, estimated_salary]],
                            columns=["CreditScore","Gender","Age","Tenure","Balance","HasCrCard","IsActiveMember","EstimatedSalary"])
    
    # Preprocess like training
        input_data = pd.get_dummies(input_df, drop_first=True)

    # Add missing columns from model
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0

    # Ensure correct order
        input_data = input_data[model_columns]

    # Predictions
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.success("✅ Not Likely to Churn" if prediction==0 else "⚠️ Likely to Churn")
        st.progress(int(proba*100))
        st.write(f"Churn Probability: {proba:.2%}")


elif option == "📂 Batch Prediction":
    st.subheader("Upload CSV for batch predictions")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("📂 Raw Data", data.head())

        input_data = preprocess_data(data, model_columns)

        predictions = model.predict(input_data)
        proba = model.predict_proba(input_data)[:, 1]

        data["Churn Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
        data["Churn Probability"] = proba

        st.write("📊 Predictions", data[["Churn Prediction", "Churn Probability"]].head())

        st.subheader("📈 Churn Distribution")
        fig = px.histogram(data, x="Churn Prediction", color="Churn Prediction", color_discrete_map={"Yes": "red", "No": "green"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⬇️ Download Predictions")
        st.download_button("Download CSV", data.to_csv(index=False).encode("utf-8"), file_name="churn_predictions.csv")

elif option == "📈 Model Insights":
    st.subheader("Feature Importance & SHAP Analysis")
    st.markdown("This plot shows the global feature importance. Features at the top have a larger impact on the model's predictions.")

    try:
        # Minimal raw input (8 features)
        raw_sample = pd.DataFrame([{
            "CreditScore": 600,
            "Gender": "Male",
            "Age": 35,
            "Tenure": 3,
            "Balance": 50000,
            "HasCrCard": 0,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000
        }])

        # Preprocess to match training features
        sample_data = preprocess_data(raw_sample, model_columns)

        # SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_data)

        # Plot SHAP summary
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, sample_data, plot_type="bar", show=False)
        st.pyplot(fig, bbox_inches="tight")

    except Exception as e:
        st.error(f"An error occurred while generating the SHAP plot: {e}")

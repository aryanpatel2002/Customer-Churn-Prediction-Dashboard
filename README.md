# Customer-Churn-Prediction-Dashboard

## 📝 Project Overview

The **Customer Churn Predictor Dashboard** is an interactive web application that predicts customer churn using a trained machine learning model. The dashboard provides intuitive visualizations, key metrics, and explanations for model predictions, helping businesses make informed decisions to retain customers.
link: https://customerchurn1o1.streamlit.app/

**Key Features:**

* Predict if a customer is likely to churn based on input features
* Interactive data visualizations using Plotly and Matplotlib
* SHAP explainability for model predictions
* User-friendly web interface powered by Streamlit
* Custom styling for a clean, modern dashboard experience

---

## 💻 Tech Stack

* **Backend & ML:** Python, scikit-learn, joblib
* **Web Framework:** Streamlit
* **Data Handling & Visualization:** Pandas, Plotly, Matplotlib, SHAP
* **Environment:** Virtualenv

---

## 📂 Folder Structure

```
customer-churn-dashboard/
│
├── app/
│   ├── app.py               # Main Streamlit application
│   ├── style.css            # Custom CSS for dashboard
│   └── utils.py             # Helper functions (load model, preprocess, SHAP explainer)
│
├── models/
│   └── churn_model.pkl      # Trained ML model
│
├── data/                    # Sample dataset (if any)
│
├── requirements.txt         # Python dependencies
│
└── README.md                # Project documentation
```

---

## ⚡ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/customer-churn-dashboard.git
cd customer-churn-dashboard
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the dashboard**

```bash
streamlit run app/app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

---

## 🚀 Usage

* Enter customer details in the input fields (e.g., tenure, services subscribed, payment method)
* Click **Predict** to see if the customer is likely to churn
* View visualizations and SHAP explanations for model insights

---

## 📊 Features & Visuals

* **Churn Probability:** Displays likelihood of churn for each customer
* **Feature Importance:** SHAP plots showing which features most influence predictions
* **Interactive Charts:** Trends and distribution analysis using Plotly

---

## 👨‍💻 Author

**Aryan Patel**

* [LinkedIn](https://www.linkedin.com/in/aryanpateldev/)

---

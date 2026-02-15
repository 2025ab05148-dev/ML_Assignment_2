import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(page_title="Breast Cancer Prediction App")

st.title("Breast Cancer Prediction Web Application")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload CSV File (Test Data Only)", type=["csv"])

# ---------------- Model Selection ----------------
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic_Regression",
        "Decision_Tree",
        "KNN",
        "Naive_Bayes",
        "Random_Forest",
        "XGBoost"
    ]
)

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # ---------------- Path Handling (Streamlit Cloud Safe) ----------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(BASE_DIR, "models")

    scaler_path = os.path.join(models_folder, "scaler.pkl")
    model_path = os.path.join(models_folder, f"{model_choice}.pkl")

    # Load scaler and model
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # ---------------- Check target column ----------------
    if "target" not in df.columns:
        st.error("❌ Uploaded CSV must contain a 'target' column for evaluation.")
        st.stop()

    # Separate features & target
    X = df.drop("target", axis=1)
    y_true = df["target"]

    # ---------------- Scaling & Prediction ----------------
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Add predictions to dataframe
    df["Prediction"] = y_pred

    st.subheader("Prediction Results")
    st.dataframe(df.head())

    # ---------------- Evaluation Metrics ----------------
    st.subheader("Evaluation Metrics")

    accuracy = accuracy_score(y_true, y_pred)
    st.write("✅ Accuracy :", round(accuracy, 4))

    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))

    # ---------------- Confusion Matrix ----------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()



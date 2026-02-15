import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page config
st.set_page_config(page_title="Breast Cancer Prediction App")
st.title("Breast Cancer Prediction Web Application")

# File upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Model selection
models_choice = st.selectbox(
    "Select Model",
    [
        "logistic",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    # Check target column
    if "target" not in df.columns:
        st.error("CSV must contain 'target' column for evaluation.")
        st.stop()

    y_true = df["target"]
    X_input = df.drop("target", axis=1)

    # Safe path handling
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(BASE_DIR, "models")

    scaler_path = os.path.join(models_folder, "scaler.pkl")
    model_path = os.path.join(models_folder, f"{models_choice}.pkl")

    # Load model and scaler
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # Scale features
    X_scaled = scaler.transform(X_input)

    # Prediction
    predictions = model.predict(X_scaled)

    df["Prediction"] = predictions

    st.subheader("Prediction Results")
    st.dataframe(df.head())

    # -------- Evaluation Metrics --------
    accuracy = accuracy_score(y_true, predictions)

    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", round(accuracy, 4))

    st.subheader("Classification Report")
    st.text(classification_report(y_true, predictions))

    # -------- Confusion Matrix --------
    cm = confusion_matrix(y_true, predictions)

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

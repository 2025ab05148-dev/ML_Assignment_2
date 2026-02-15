import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page config
st.set_page_config(page_title="Stroke Prediction App")
st.title("Stroke Prediction Web Application")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

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

    # -------- Target Handling --------
    if "stroke" not in df.columns:
        st.error("CSV must contain 'stroke' column.")
        st.stop()

    y_true = df["stroke"]
    X_input = df.drop("stroke", axis=1)

    # Drop id column
    if "id" in X_input.columns:
        X_input = X_input.drop("id", axis=1)

    # -------- Handle Categorical Columns --------
    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type"]

    for col in categorical_cols:
        if col in X_input.columns:
            X_input[col] = X_input[col].astype(str)

    X_input = pd.get_dummies(X_input)

    # -------- Model Loading --------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(BASE_DIR, "models")

    scaler_path = os.path.join(models_folder, "scaler.pkl")
    model_path = os.path.join(models_folder, f"{models_choice}.pkl")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # -------- Scaling --------
    X_scaled = scaler.transform(X_input)

    # -------- Prediction --------
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

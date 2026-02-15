import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(page_title="Breast Cancer Prediction App")

st.title("Breast Cancer Prediction Web Application")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

models_choice = st.selectbox(
    "Select models",
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

    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())

    # Safe path handling for Streamlit Cloud
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(BASE_DIR, "models")

    scaler_path = os.path.join(models_folder, "scaler.pkl")
    models_path = os.path.join(models_folder, f"{models_choice}.pkl")

    scaler = joblib.load(scaler_path)
    models = joblib.load(models_path)

    X = scaler.transform(df)
    predictions = models.predict(X)

    df["Prediction"] = predictions

    st.subheader("Prediction Results")
    st.dataframe(df)


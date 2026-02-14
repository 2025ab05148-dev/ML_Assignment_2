import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(page_title="Breast Cancer Prediction App")

st.title("Breast Cancer Prediction Web Application")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

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

    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())

    # Safe path handling for Streamlit Cloud
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(BASE_DIR, "model")

    scaler_path = os.path.join(model_folder, "scaler.pkl")
    model_path = os.path.join(model_folder, f"{model_choice}.pkl")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    X = scaler.transform(df)
    predictions = model.predict(X)

    df["Prediction"] = predictions

    st.subheader("Prediction Results")
    st.dataframe(df)


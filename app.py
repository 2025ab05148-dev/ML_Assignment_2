import streamlit as st
import pandas as pd
import joblib

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

    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load(f"models/{model_choice}.pkl")

    X = scaler.transform(df)
    predictions = model.predict(X)

    df["Stroke_Prediction"] = predictions

    st.subheader("Prediction Results")
    st.dataframe(df)

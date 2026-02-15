import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Stroke Prediction App")
st.title("Stroke Prediction Web Application")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

models_choice = st.selectbox(
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
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    if "stroke" not in df.columns:
        st.error("CSV must contain 'stroke' column.")
        st.stop()

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    y_true = df["stroke"]
    X_input = df.drop("stroke", axis=1)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_folder = os.path.join(BASE_DIR, "models")

    scaler = joblib.load(os.path.join(models_folder, "scaler.pkl"))
    model = joblib.load(os.path.join(models_folder, f"{models_choice}.pkl"))

    X_scaled = scaler.transform(X_input)
    predictions = model.predict(X_scaled)

    df["Prediction"] = predictions

    st.subheader("Prediction Results")
    st.dataframe(df.head())

    accuracy = accuracy_score(y_true, predictions)

    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", round(accuracy, 4))

    st.subheader("Classification Report")
    st.text(classification_report(y_true, predictions))

    cm = confusion_matrix(y_true, predictions)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "Training.csv"
MODEL_PATH = "model.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"


@st.cache_resource
def train_and_save_model():
    data = pd.read_csv(DATA_PATH)
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    symptoms = [col for col in data.columns if col != "prognosis"]
    X = data[symptoms]
    y = data["prognosis"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    return symptoms, model, label_encoder, accuracy


st.set_page_config(page_title="Disease Prediction App", page_icon="🩺", layout="wide")
st.title("Disease Prediction and Diagnosis")
st.markdown(
    "This app predicts the most likely disease based on the symptoms selected from the training dataset."
)

symptoms, model, label_encoder, accuracy = train_and_save_model()

st.info(f"Model accuracy on the validation split: {accuracy:.2%}")

with st.container():
    st.subheader("Enter the patient symptoms")
    symptom_values = {}
    for symptom in symptoms:
        symptom_values[symptom] = st.checkbox(symptom, key=symptom)

    input_vector = np.array(
        [1 if symptom_values[symptom] else 0 for symptom in symptoms], dtype=int
    ).reshape(1, -1)

    if st.button("Predict Disease"):
        predicted_index = model.predict(input_vector)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_index])[0]
        st.success(f"Predicted disease: {predicted_disease}")

st.caption("Model and label encoder are saved to the project folder for reuse.")

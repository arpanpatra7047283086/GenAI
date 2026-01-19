import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import os



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "Saved_Model", "churn_model.h5")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "Saved_Model", "preprocessor.pkl")


if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.write("Expected path:", MODEL_PATH)
    st.write("Files in project root:", os.listdir(BASE_DIR))
    st.stop()

if not os.path.exists(PREPROCESSOR_PATH):
    st.error(" Preprocessor file not found!")
    st.stop()


model = tf.keras.models.load_model(MODEL_PATH)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)


st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 40)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", value=60000.0)
num_of_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)


input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})


input_processed = preprocessor.transform(input_data)

prediction = model.predict(input_processed)
prediction_proba = float(prediction[0][0])

# -----------------------------
# OUTPUT
# -----------------------------
st.subheader(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.error("⚠️ The customer is likely to churn")
else:
    st.success("✅ The customer is not likely to churn")

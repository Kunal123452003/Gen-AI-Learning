import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("salary_reg_model.h5")

# load the encoder
with open("Gen.pkl","rb") as file:
    gen_encoder = pickle.load(file)

with open("Geo.pkl","rb") as file:
    geo_encoder = pickle.load(file)

with open("Scaler.pkl","rb") as file:
    scaler = pickle.load(file)

## Streamlit app
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", geo_encoder.categories_[0])  # to get the category
gender = st.selectbox("Gender", gen_encoder.classes_)
age = st.slider("Age", min_value=18, max_value=88)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")

tenure = st.slider("Tenure", min_value=0, max_value=10)
num_of_products = st.slider("Number of Products", max_value=4, min_value=1)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
is_exited = st.selectbox("Exited", ["Yes", "No"])

# Prepare the input data
input_data = pd.DataFrame({

    "CreditScore": [credit_score],
    "Gender":[gen_encoder.transform([gender])[0]],

    "Age":[age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_cr_card == "Yes" else 0],
    "IsActiveMember": [1 if is_active_member == "Yes" else 0],
    "Exited": [1 if is_exited == "Yes" else 0]
})

geo_encoded = geo_encoder.transform([[geography]]).reshape(1,-1)
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

# Concatenate the geo_encoded_df with input_data
input_data= pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the numerical features
input_data = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data)
prediction_probability = prediction[0][0]

st.write(f"Prediction Probability: {prediction_probability:.2f}")

print(f"Prediction Salary: {prediction:}")


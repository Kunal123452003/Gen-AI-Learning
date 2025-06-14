import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
def load_dataset():
    diabetes = load_diabetes(scaled=False)
    df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    return df

# Train a simple model
df = load_dataset()

X = df.drop("target", axis=1)
y = df["target"]

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Train a Random Forest Classifier
model = RandomForestRegressor()
model.fit(X_scaled,y)

# sidebar ,slider, radio for selecting a feature
st.sidebar.title("Features Selection")
age = st.sidebar.slider("Age", min_value=int(X['age'].min()), max_value=int(X['age'].max()))
sex = st.sidebar.radio("Sex", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
bmi = st.sidebar.slider("BMI", min_value=int(X['bmi'].min()), max_value=int(X['bmi'].max()))
bp = st.sidebar.slider("BP", min_value=int(X['bp'].min()), max_value=int(X['bp'].max()))
s1 = st.sidebar.slider("Total serum cholesterol", min_value=int(X['s1'].min()), max_value=int(X['s1'].max()))
s2 = st.sidebar.slider("Low-density lipoproteins", min_value=int(X['s2'].min()), max_value=int(X['s2'].max()))
s3= st.sidebar.slider("High-density lipoproteins", min_value=int(X['s3'].min()), max_value=int(X['s3'].max()))
s4 = st.sidebar.slider("Total cholesterol / HDL", min_value=int(X['s4'].min()), max_value=int(X['s4'].max()))
s5 = st.sidebar.slider("Possibly log of serum triglycerides level", min_value=float(X['s5'].min()), max_value=float(X['s5'].max()))
s6 = st.sidebar.slider("Blood sugar level", float(X['s6'].min()), float(X['s6'].max()))


predict_data = np.array([[age,sex,bmi,bp,s1,s2,s3,s4,s5,s6]])
# st.write("predict_data dim:", predict_data.shape)

predict_data = sc.transform(predict_data)
# st.write("Scaled predict_data dim:", predict_data.shape)

st.title("Predicted Value:")
st.write(model.predict(predict_data)[0])

# a = np.array([[age,sex,bmi,bp,s1,s2,s3,s4,s5,s6]]).reshape(1,-1)
# st.write(a.shape)
# st.write(a)
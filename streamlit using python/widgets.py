import streamlit as st
import pandas as pd

st.title("streamlit widgets")

#Text input widget 
st.title("Text Input")
name = st.text_input("Enter your name:")

if name:
    st.write(f"Hello {name}")

# Number input widget
age = st.number_input("Enter your age:", min_value=0, max_value=200, value=0, step=1 )
if age:
    st.write(f"You are {age} years old")

# slider widget
st.write("Year of experience:",divider="horizontal")
exp = st.slider("Select year of experience", 0,20, 1) # 0 to 20 is range and default is 1
st.write(f"Experience : {exp} yrs ")

# Selectbox widget
options = ["Python", "Java", "C++", "JavaScript"]
choice = st.selectbox("Select a programming language:",options)
st.write(f"You selected: {choice}")

# upload file widget
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}

df = pd.DataFrame(data)
df.to_csv("Sample_data.csv")
st.write(df)

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    Uploaded_df = pd.read_csv(uploaded_file)
    st.write(Uploaded_df)
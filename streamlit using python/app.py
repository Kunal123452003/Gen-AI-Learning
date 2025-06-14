""" Streamlit allow us to create web application using python """

import streamlit as st
import pandas as pd
import numpy as np

# Title of the app
st.title("Title of the app")

# Display simple text
st.write("This is a simple text")

# Display a dataframe
df = pd.DataFrame({
    "Column 1": [1, 2, 3, 4],
    "Column 2": [10, 20, 30, 40],
    "Column 3": [100, 200, 300, 400]
})
st.write("This is a dataframe")
st.write(df)

# Display a chart
chart_data = pd.DataFrame(
    np.random.randn(20,3), columns=["a","b","c"]
)
st.write("This is Chart data")
st.write(chart_data)

st.write("Chart using line chart")
st.line_chart(chart_data)
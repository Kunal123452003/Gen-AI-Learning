# 1) importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
import streamlit as st


# 2) load word index from imdb dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# 3) Load the saved model
model = load_model('SimpleRNN_imdb_improved.keras')

# Step 2: helper func
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

# func to process text
def preprocess_text(text):
    print("Original review:", text)
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) +3 for word in words]
    padded_review =sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
    
# Step 3 : prediction function
## prediction fucntion
def predict_sentiment(text_review):
    # preprocessed_input = preprocess_text(text_review)
    prediction = model.predict(text_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]

# step 4: Streamlit app
st.title("IMDB movie review sentiment analysis")
st.write("Enter a movie review to predict it as positive or negative.")

# User input
user_input = st.text_area("movie review")
print("User input:", user_input)

if st.button("Sentiment"):

    preprocess_input = preprocess_text(user_input)
    st.write("Preprocessed input:", preprocess_input)

    sentiment, prediction = predict_sentiment(preprocess_input)
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction} ")

else:
    st.write("Please enter a movie review and click the button to predict its sentiment.")


# 1) importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# 2) load word index from imdb dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# 3) Load the saved model
model = load_model("SimpleRNN_imdb.h5")

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3 ,'?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3  for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=250)
    return padded_review

# 4) Prediction function
def predict_sentiment(text_review):
    preprocessed_input = preprocess_text(text_review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]

# streamlit for web app
import streamlit as st

# streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative:")

# User input
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    preprocessed_input = preprocess_text

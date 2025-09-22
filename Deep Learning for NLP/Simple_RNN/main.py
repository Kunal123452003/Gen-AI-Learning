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

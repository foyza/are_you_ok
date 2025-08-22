import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle


data = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)


word_index = data.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(text_ids):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in text_ids])

max_length = 200
train_data = pad_sequences(train_data, maxlen=max_length)
test_data = pad_sequences(test_data, maxlen=max_length)


model = Sequential([
    Embedding(10000, 16),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(train_data, train_labels, epochs=2, batch_size=512, validation_split=0.2)


model.save("model.h5")


with open("tokenizer.pkl", "wb") as f:
    pickle.dump(word_index, f)


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.DataFrame({
    "text": ["я счастлив", "я грустный", "жизнь прекрасна", "мне плохо"],
    "label": [1, 0, 1, 0]
})

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(data["text"])
X = tokenizer.texts_to_sequences(data["text"])
X = pad_sequences(X, maxlen=10)
y = np.array(data["label"])

model = Sequential([
    Embedding(1000, 16, input_length=10),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10, verbose=0)
model.save("model.h5")
print("✅ Модель сохранена")

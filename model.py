import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 64
DATASET_DIR = "dataset/images"
LABELS_FILE = "dataset/labels.csv"

labels_df = pd.read_csv(LABELS_FILE)
X, y = [], []
for idx, row in labels_df.iterrows():
    img_path = os.path.join(DATASET_DIR, row['filename'])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    X.append(img)
    y.append(row['happiness'])
X = np.array(X)
y = np.array(y)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)
model.save("model.h5")
print("Model saved as model.h5")

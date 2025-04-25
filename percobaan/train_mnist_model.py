import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("dataset/sign_mnist_train.csv")
X = df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y = to_categorical(df["label"])

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(25, activation='softmax')  # 25 huruf (A–Z, skip J dan Z)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.1)

MODEL_DIR = '/model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Simpan model
model.save("model/sign_mnist_cnn.h5")
print("Model disimpan ke model/sign_mnist_cnn.h5")

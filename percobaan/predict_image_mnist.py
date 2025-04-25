import sys
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

# Mapping ke huruf A–Z, skip J dan Z
label_map = {i: chr(ord('A') + i + (1 if i >= 9 else 0)) for i in range(24)}  # 0–23 mewakili A–Y (skip J & Z)

def preprocess_image(image_path):
    img = cv2.imread(image_path) # Baca gambar warna
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert ke grayscale
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1) / 255.0
    return img

MODEL_FILE = 'model/sign_mnist_cnn.h5'
if not os.path.exists(MODEL_FILE):
    print('FIle Model tidak ada.')
    exit()

# model = pd.
model = load_model("model/sign_mnist_cnn.h5")
image_path = sys.argv[1]
img = preprocess_image(image_path)
pred = model.predict(img)
predicted_class = np.argmax(pred)

# Adjust index for skipped labels
adjusted_idx = predicted_class
for skip in [9, 25]:
    if predicted_class >= skip:
        adjusted_idx += 1

print("Prediksi huruf:", chr(ord('A') + adjusted_idx))


df = pd.read_csv("dataset/sign_mnist_train.csv")
sample = df[df['label'] == 4].iloc[0, 1:].values.reshape(28,28)  # label 0 = A

plt.imshow(sample, cmap='gray')
plt.show()
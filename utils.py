import os
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = 'model/brain_trauma_model.h5'
MODEL_URL = 'model\brain_trauma_model.h5'  # <-- replace this

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading model from:", MODEL_URL)
        r = requests.get(MODEL_URL, stream=True)
        if r.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            raise Exception(f"Failed to download model: {r.status_code}")

# Download and load the model once
download_model()
model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    print("Prediction Probability:", prediction[0][0])

    if prediction[0][0] > 0.5:
        return "Trauma Detected"
    else:
        return "Normal"

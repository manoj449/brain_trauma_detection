import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('model/brain_trauma_model.h5')  # Ensure this is trained properly

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    print("Prediction Probability:", prediction[0][0])  # Add this line

    if prediction[0][0] > 0.5:
        return "Trauma Detected"
    else:
        return "Normal"


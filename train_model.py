import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define paths and parameters
TRAIN_DIR = 'brain_mri_synthetic'  # Path to extracted dataset
IMG_SIZE = 224  # Image dimensions
BATCH_SIZE = 32  # Batch size for training

# Data generator for loading images from directories
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize image data

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),  # Resize images to 224x224
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Binary classification (trauma or no trauma)
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid output for binary classification
])

# Compile the model
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    epochs=10,  # Number of epochs, you can increase this
    steps_per_epoch=train_gen.samples // BATCH_SIZE
)

# Save the trained model
model.save('brain_trauma_model.h5')

print("✅ Model trained and saved as 'brain_trauma_model.h5'")

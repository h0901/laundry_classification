import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class CNN:
    def __init__(self, input_shape=(64, 64, 1), num_classes=10):
        """Initialize CNN model."""
        self.model = self.build_cnn(input_shape, num_classes)
        
    def build_cnn(self, input_shape, num_classes):
        """Build a CNN classifier."""
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self, x_train, y_train, x_test, y_test, epochs=10):
        history = self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        return history  # Ensure this is returned

    def save_model(self, path='saved_models/cnn'):
        """Save the trained model."""
        self.model.save(path)

    def load_model(self, path='saved_models/cnn'):
        """Load a pre-trained model."""
        self.model = tf.keras.models.load_model(path)

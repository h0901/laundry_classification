import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class TransformerModel:
    def __init__(self, input_shape=(64, 64, 1), num_classes=10, patch_size=8, num_patches=64, projection_dim=64, num_heads=4, transformer_units=[128, 64]):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.model = self.build_model()

    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # Patching
        patches = layers.Conv2D(filters=self.projection_dim,
                                kernel_size=self.patch_size,
                                strides=self.patch_size,
                                padding='VALID')(inputs)
        patches = layers.Reshape((self.num_patches, self.projection_dim))(patches)

        # Positional Encoding
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embedding = layers.Embedding(input_dim=self.num_patches, output_dim=self.projection_dim)(positions)
        encoded_patches = patches + position_embedding

        # Transformer Block
        for _ in range(1):  # One transformer block for simplicity
            # Layer Norm + Multi-head Attention
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim)(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])

            # Feed-forward + residual
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])

        # Classification head
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        features = layers.Dense(128, activation="relu")(representation)
        logits = layers.Dense(self.num_classes, activation="softmax")(features)

        model = models.Model(inputs=inputs, outputs=logits)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train(self, x_train, y_train, x_test, y_test, epochs=10):
        history = self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        return history

    def save_model(self, path='saved_models/transformer.h5'):
        self.model.save(path)

    def load_model(self, path='saved_models/transformer'):
        self.model = tf.keras.models.load_model(path)

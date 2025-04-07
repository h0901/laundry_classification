import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class BEGAN:
    def __init__(self, latent_dim=100, img_shape=(64, 64, 1), num_classes=10):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes

        # Build the model
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.combined = self.build_combined()

        # Compile the models
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.generator.compile(loss='binary_crossentropy', optimizer='adam')
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')

    def build_generator(self):
        noise = layers.Input(shape=(self.latent_dim,))
        label_input = layers.Input(shape=(1,))  # Label input

        # Embed label input to match the latent space
        label_embedding = layers.Embedding(self.num_classes, self.latent_dim)(label_input)
        label_embedding = layers.Flatten()(label_embedding)

        # Concatenate the noise vector and the label embedding
        combined_input = layers.Concatenate()([noise, label_embedding])

        # Define the generator layers
        x = layers.Dense(256, activation='relu')(combined_input)
        x = layers.Reshape((16, 16, 1))(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = layers.UpSampling2D()(x)
        gen_img = layers.Conv2D(self.img_shape[2], kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

        # Create the generator model
        model = models.Model([noise, label_input], gen_img)
        return model

    def build_discriminator(self):
        img_input = layers.Input(shape=self.img_shape)
        label_input = layers.Input(shape=(1,))  # Label input

        # Embed label input to match image input
        label_embedding = layers.Embedding(self.num_classes, np.prod(self.img_shape))(label_input)
        label_embedding = layers.Reshape(self.img_shape)(label_embedding)

        # Concatenate image input with the label embedding
        combined_input = layers.Concatenate()([img_input, label_embedding])

        # Define the discriminator layers
        x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(combined_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Flatten()(x)
        validity = layers.Dense(1, activation='sigmoid')(x)

        # Create the discriminator model
        model = models.Model([img_input, label_input], validity)
        return model

    def build_combined(self):
        noise = layers.Input(shape=(self.latent_dim,))
        label_input = layers.Input(shape=(1,))

        # Generate an image using the generator
        gen_img = self.generator([noise, label_input])

        # Discriminate if the generated image is real or fake
        validity = self.discriminator([gen_img, label_input])

        # Create the combined model
        model = models.Model([noise, label_input], validity)
        return model

    def generate_samples(self, num_samples=1):
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to 0-1
        return gen_imgs

    # Save models to files
    def save_models(self, model_dir):
        self.generator.save(f'{model_dir}/generator.h5')
        self.discriminator.save(f'{model_dir}/discriminator.h5')
        print(f"Models saved to {model_dir}")
    
    # Load models from files
    def load_models(self, model_dir):
        self.generator = tf.keras.models.load_model(f'{model_dir}/generator.h5')
        self.discriminator = tf.keras.models.load_model(f'{model_dir}/discriminator.h5')
        print(f"Models loaded from {model_dir}")

# Helper function to load BEGAN model
def load_began(model_path):
    began = BEGAN(latent_dim=100, img_shape=(64, 64, 3))
    began.load_models(model_path)
    return began

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os

class ACGAN:
    def __init__(self, img_shape=(64, 64, 1), num_classes=10, latent_dim=100):
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
            optimizer=optimizers.Adam(0.0002, 0.5),
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = self.build_generator()
        
        # Combined model (stacked generator and discriminator)
        self.combined = self.build_combined()
        self.combined.compile(
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
            optimizer=optimizers.Adam(0.0002, 0.5)
        )
        
    def build_generator(self):
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(1,), dtype='int32')
        
        # Embed label into the latent space
        label_embedding = layers.Embedding(self.num_classes, self.latent_dim)(label)
        label_embedding = layers.Flatten()(label_embedding)
        
        # Combine noise and label
        model_input = layers.multiply([noise, label_embedding])
        
        # Foundation for 8x8 image
        x = layers.Dense(128 * 8 * 8, activation="relu")(model_input)
        x = layers.Reshape((8, 8, 128))(x)
        
        # Upsample to 16x16
        x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        # Upsample to 32x32
        x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        # Upsample to 64x64
        x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        # Output layer
        img = layers.Conv2D(self.img_shape[2], 3, padding="same", activation="tanh")(x)
        
        return models.Model([noise, label], img, name="generator")
    
    def build_discriminator(self):
        img = layers.Input(shape=self.img_shape)
        
        # Downsample
        x = layers.Conv2D(32, 3, strides=2, padding="same")(img)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Flatten()(x)
        
        # Two output layers:
        # 1. Validity (real/fake)
        validity = layers.Dense(1, activation="sigmoid")(x)
        # 2. Class label
        label = layers.Dense(self.num_classes, activation="softmax")(x)
        
        return models.Model(img, [validity, label], name="discriminator")
    
    def build_combined(self):
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(1,))
        img = self.generator([noise, label])
        
        self.discriminator.trainable = False
        valid, target_label = self.discriminator(img)
        
        return models.Model([noise, label], [valid, target_label], name="ACGAN")
    
    def generate_samples(self, num_samples=1, class_label=None):
        if class_label is None:
            class_label = np.random.randint(0, self.num_classes, num_samples).reshape(-1, 1)
        else:
            class_label = np.full((num_samples, 1), class_label)
        
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        gen_imgs = self.generator.predict([noise, class_label])
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to 0-1
        
        return gen_imgs
    
    def save_models(self, path="saved_models"):
        os.makedirs(path, exist_ok=True)
        self.generator.save(f"{path}/acgan_generator.h5")
        self.discriminator.save(f"{path}/acgan_discriminator.h5")
        self.combined.save(f"{path}/acgan_combined.h5")
    
    def load_models(self, path="saved_models"):
        self.generator = models.load_model(f"{path}/acgan_generator.h5")
        self.discriminator = models.load_model(f"{path}/acgan_discriminator.h5")
        self.combined = models.load_model(f"{path}/acgan_combined.h5")

# Helper function to load ACGAN model
def load_gan(model_path, gan_type='acgan'):
    if gan_type != 'acgan':
        raise ValueError("This function only loads ACGAN models")
    
    acgan = ACGAN()
    acgan.load_models(model_path)
    return acgan

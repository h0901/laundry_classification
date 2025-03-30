import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DCGAN:
    def __init__(self, img_shape=(64, 64, 1), latent_dim=100):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(0.0002, 0.5),
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = self.build_generator()
        
        # Combined model (stacked generator and discriminator)
        self.combined = self.build_combined()
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(0.0002, 0.5)
        )
        
    def build_generator(self):
        noise = layers.Input(shape=(self.latent_dim,))
        
        # Dense layer to reshape the input to an image
        x = layers.Dense(128 * 8 * 8, activation="relu")(noise)
        x = layers.Reshape((8, 8, 128))(x)
        
        # Upsample to 64x64 image
        x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        
        img = layers.Conv2D(self.img_shape[2], 3, padding="same", activation="tanh")(x)
        
        return models.Model(noise, img, name="generator")
    
    def build_discriminator(self):
        img = layers.Input(shape=self.img_shape)
        
        x = layers.Conv2D(32, 3, strides=2, padding="same")(img)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Flatten()(x)
        validity = layers.Dense(1, activation="sigmoid")(x)
        
        return models.Model(img, validity, name="discriminator")
    
    def build_combined(self):
        noise = layers.Input(shape=(self.latent_dim,))
        img = self.generator(noise)
        
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        
        return models.Model(noise, validity, name="DCGAN")
    
    def generate_samples(self, num_samples=1):
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to 0-1
        return gen_imgs

    def save_models(self, path="saved_models"):
        os.makedirs(path, exist_ok=True)
        self.generator.save(f"{path}/dcgan_generator.h5")
        self.discriminator.save(f"{path}/dcgan_discriminator.h5")
        self.combined.save(f"{path}/dcgan_combined.h5")

    def load_models(self, path="saved_models"):
        self.generator = models.load_model(f"{path}/dcgan_generator.h5")
        self.discriminator = models.load_model(f"{path}/dcgan_discriminator.h5")
        self.combined = models.load_model(f"{path}/dcgan_combined.h5")

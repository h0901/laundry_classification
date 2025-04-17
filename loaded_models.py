import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from visualize import compare_gan_heatmaps, compare_generated_images, plot_accuracy_heatmap, plot_full_history_comparison

import sys
sys.path.append("./gans")
from acgan import ACGAN
from dcgan import DCGAN
from wgan import WGAN
from began import BEGAN

acgan = ACGAN().load_models("saved_models")
dcgan = DCGAN().load_models("saved_models")
began = BEGAN().load_models("saved_models")
wgan = WGAN().load_models("saved_models")

compare_generated_images(acgan, dcgan, began, wgan)

gans = {
    'ACGAN': acgan,
    'DCGAN': dcgan,
    'WGAN': wgan,
    'BEGAN': began
}

compare_gan_heatmaps(gans, num_images=6)

class GAN:
    def __init__(self, generator_path, img_shape=(64, 64, 1), latent_dim=100, num_classes=10):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Load the generator model
        self.generator = load_model(generator_path)
        
    def generate_samples(self, num_samples=1, class_label=None):
        if class_label is None:
            class_label = np.random.randint(0, self.num_classes, num_samples).reshape(-1, 1)
        else:
            class_label = np.full((num_samples, 1), class_label)
        
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        gen_imgs = self.generator.predict([noise, class_label])
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to 0-1
        
        return gen_imgs

    def visualize_generated_images(self, num_samples=16):
        gen_imgs = self.generate_samples(num_samples)
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(4):
            for j in range(4):
                ax = axes[i, j]
                ax.imshow(gen_imgs[i * 4 + j].reshape(64, 64), cmap='gray')  # Adjust reshape based on image size
                ax.axis('off')
        plt.show()

# Helper function to load and visualize ACGAN, DCGAN, WGAN, BEGAN
def load_and_visualize_gan(model_path, gan_type='acgan'):
    # Initialize GAN model
    gan = GAN(generator_path=model_path)
    
    # Visualize generated images
    gan.visualize_generated_images(num_samples=16)  # Visualize 16 images in a 4x4 grid

# Example usage for all GAN models
acgan_model_path = "saved_models/acgan_generator.h5"
dcgan_model_path = "saved_models/dcgan_generator.h5"
wgan_model_path = "saved_models/wgan_generator.h5"
began_model_path = "saved_models/began_generator.h5"

print("Visualizing ACGAN:")
load_and_visualize_gan(acgan_model_path)

print("Visualizing DCGAN:")
load_and_visualize_gan(dcgan_model_path)

print("Visualizing WGAN:")
load_and_visualize_gan(wgan_model_path)

print("Visualizing BEGAN:")
load_and_visualize_gan(began_model_path)

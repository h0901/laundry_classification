import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Embedding, Concatenate
from tensorflow.keras import Model

# Helper function to load models
def load_gan_model(generator_path, discriminator_path):
    generator = load_model(generator_path)
    discriminator = load_model(discriminator_path)
    return generator, discriminator

# Example function to generate fake images and corresponding labels
def generate_fake_images(generator, latent_dim, num_samples=100, num_classes=10):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))  # Latent space noise
    labels = np.random.randint(0, num_classes, num_samples)  # Random labels for class conditioning
    
    # Generate fake images using the generator
    fake_images = generator.predict([noise, labels])
    
    # Normalize images to [0, 1] range
    fake_images = 0.5 * fake_images + 0.5
    
    return fake_images, labels  # Return two variables: fake images and labels

def evaluate_model_output(model, fake_images, labels, is_critic=False):
    if is_critic:
        # For WGAN, we take the mean of the critic's output (since it's continuous)
        critic_output = model.predict(fake_images)
        mean_critic_output = np.mean(critic_output)
        return mean_critic_output  # Return the mean critic output as a measure
    else:
        # For other models, evaluate the discriminator (binary classification)
        
        # Embed the labels and concatenate them with the fake images
        # If the discriminator expects the image and label together, we concatenate them
        label_embedding = np.expand_dims(labels, axis=-1)  # Expand dimensions of labels to match image shape
        
        # Concatenate images and labels along the channel axis (axis=-1)
        fake_images_with_labels = [fake_images, label_embedding]  # Modify this according to the discriminator's input
        
        fake_labels = np.zeros((fake_images.shape[0], 1))  # Fake images labeled as 0 (fake)
        accuracy = model.evaluate(fake_images_with_labels, fake_labels, verbose=0)  # Pass labels to discriminator
        return accuracy[1]  # Return accuracy (index 1 corresponds to accuracy)

# Helper function to plot the performance of all GAN models
def plot_performance(accuracies, model_names):
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple'])
    plt.title('Discriminator and Critic Performance Comparison')
    plt.xlabel('GAN Models')
    plt.ylabel('Discriminator/ Critic Output')
    plt.grid(True)
    plt.show()

# Paths to your generator and discriminator models
acgan_gen_path = "saved_models/acgan_generator.h5"
acgan_disc_path = "saved_models/acgan_discriminator.h5"

dcgan_gen_path = "saved_models/dcgan_generator.h5"
dcgan_disc_path = "saved_models/dcgan_discriminator.h5"

wgan_gen_path = "saved_models/wgan_generator.h5"
wgan_critic_path = "saved_models/wgan_critic.h5"

began_gen_path = "saved_models/began_generator.h5"
began_disc_path = "saved_models/began_discriminator.h5"

# List of model names for labeling
model_names = ["ACGAN", "DCGAN", "WGAN", "BEGAN"]
gen_paths = [acgan_gen_path, dcgan_gen_path, wgan_gen_path, began_gen_path]
disc_critic_paths = [acgan_disc_path, dcgan_disc_path, wgan_critic_path, began_disc_path]

accuracies = []

# Main code execution
for gen_path, disc_critic_path in zip(gen_paths, disc_critic_paths):
    generator, model = load_gan_model(gen_path, disc_critic_path)
    
    # If the generator has multiple inputs (e.g., noise + labels), handle it
    if isinstance(generator.input, list):
        latent_dim = generator.input[0].shape[1]  # Latent space dimension from the first input (noise)
    else:
        latent_dim = generator.input.shape[1]  # For models with single input (e.g., DCGAN)
    
    # Generate fake images and labels
    fake_images, labels = generate_fake_images(generator, latent_dim)  # Pass latent_dim to the function
    
    # Evaluate the discriminator or critic on the fake images
    if 'critic' in disc_critic_path:  # Check if the model is a critic (WGAN)
        output = evaluate_model_output(model, fake_images, labels, is_critic=True)
        accuracies.append(output)  # Store critic's mean output
    else:
        # For other models, use discriminator
        accuracy = evaluate_model_output(model, fake_images, labels, is_critic=False)
        accuracies.append(accuracy)  # Store discriminator's accuracy

# Plot the performance comparison
plot_performance(accuracies, model_names)

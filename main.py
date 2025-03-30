import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cnn import CNN
from visualize import plot_training_history, visualize_generated_images  # Importing the visualization functions

import sys
sys.path.append("./gans")
from acgan import ACGAN

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize images to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Resize images to 64x64
x_train = np.expand_dims(x_train, -1)  # Add the channel dimension (1 for grayscale)
x_test = np.expand_dims(x_test, -1)

x_train = tf.image.resize(x_train, (64, 64))
x_test = tf.image.resize(x_test, (64, 64))

# Now x_train and x_test have shape (None, 64, 64, 1)

# Train original CNN
cnn = CNN()
history_cnn = cnn.train(x_train, y_train, x_test, y_test, epochs=10)

# Create an instance of ACGAN for data augmentation
acgan = ACGAN()

# Function to train ACGAN
def train_gan(gan, x_train, y_train, epochs=10000, batch_size=64):
    half_batch = batch_size // 2
    augmented_images = []
    augmented_labels = []
    
    for epoch in range(epochs):
        # Select random half batch of real images
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        
        # Convert idx to tf.int32 tensor for TensorFlow compatibility
        idx = tf.convert_to_tensor(idx, dtype=tf.int32)
        
        # Use tf.gather to index x_train and y_train
        imgs, labels = tf.gather(x_train, idx), tf.gather(y_train, idx)
        
        # Generate random noise and labels for fake images
        noise = np.random.normal(0, 1, (half_batch, gan.latent_dim))
        gen_labels = np.random.randint(0, gan.num_classes, half_batch)
        gen_imgs = gan.generator.predict([noise, gen_labels])
        
        # Train the discriminator on real and fake images
        d_loss_real = gan.discriminator.train_on_batch(imgs, [np.ones((half_batch, 1)), labels])
        d_loss_fake = gan.discriminator.train_on_batch(gen_imgs, [np.zeros((half_batch, 1)), gen_labels])
        
        # Train the generator (via the combined model)
        noise = np.random.normal(0, 1, (batch_size, gan.latent_dim))
        sampled_labels = np.random.randint(0, gan.num_classes, batch_size)
        g_loss = gan.combined.train_on_batch([noise, sampled_labels], [np.ones((batch_size, 1)), sampled_labels])
        
        # Store augmented data for later training the CNN
        augmented_images.extend(gen_imgs)  # Use extend to append the list of generated images
        augmented_labels.extend(gen_labels)  # Similarly for labels
        
        # Print the losses at regular intervals
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {0.5 * np.add(d_loss_real, d_loss_fake)}, G Loss: {g_loss}")
    
    # Convert augmented data to numpy arrays after training is done
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels

# Train ACGAN and get augmented data
augmented_images, augmented_labels = train_gan(acgan, x_train, y_train)

# Combine the original and augmented data
augmented_x_train = np.concatenate((x_train, augmented_images))
augmented_y_train = np.concatenate((y_train, augmented_labels))

# Train CNN with augmented data
cnn_augmented = CNN()
history_cnn_augmented = cnn_augmented.train(augmented_x_train, augmented_y_train, x_test, y_test, epochs=10)

# Visualize training history (accuracy and loss comparison)
plot_training_history(history_cnn, history_cnn_augmented)

# Visualize a few generated images
visualize_generated_images(acgan)

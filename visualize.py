import matplotlib.pyplot as plt
import numpy as np

# Function to plot training history (accuracy and loss comparison)
def plot_training_history(history_cnn, history_cnn_augmented):
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history_cnn.history['accuracy'], label='Original CNN Accuracy')
    plt.plot(history_cnn_augmented.history['accuracy'], label='Augmented CNN Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history_cnn.history['loss'], label='Original CNN Loss')
    plt.plot(history_cnn_augmented.history['loss'], label='Augmented CNN Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to visualize a few generated images
def visualize_generated_images(acgan, num_samples=5):
    # Generate random noise and labels
    noise = np.random.normal(0, 1, (num_samples, acgan.latent_dim))
    labels = np.random.randint(0, acgan.num_classes, num_samples).reshape(-1, 1)
    
    # Generate images
    generated_images = acgan.generator.predict([noise, labels])
    generated_images = 0.5 * generated_images + 0.5  # Rescale to 0-1

    # Plot the generated images
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

def plot_full_history_comparison(history_normal, history_augmented, history_pruned_augmented, gan_name):
    plt.figure(figsize=(14, 6))

    # Accuracy Comparison
    plt.subplot(1, 2, 1)
    plt.plot(history_normal.history['accuracy'], label='Normal Train')
    plt.plot(history_normal.history['val_accuracy'], label='Normal Val')
    plt.plot(history_augmented.history['accuracy'], label=f'{gan_name} Train')
    plt.plot(history_augmented.history['val_accuracy'], label=f'{gan_name} Val')
    plt.plot(history_pruned_augmented.history['accuracy'], label=f'{gan_name} Pruned Train')
    plt.plot(history_pruned_augmented.history['val_accuracy'], label=f'{gan_name} Pruned Val')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Comparison
    plt.subplot(1, 2, 2)
    plt.plot(history_normal.history['loss'], label='Normal Train')
    plt.plot(history_normal.history['val_loss'], label='Normal Val')
    plt.plot(history_augmented.history['loss'], label=f'{gan_name} Train')
    plt.plot(history_augmented.history['val_loss'], label=f'{gan_name} Val')
    plt.plot(history_pruned_augmented.history['loss'], label=f'{gan_name} Pruned Train')
    plt.plot(history_pruned_augmented.history['val_loss'], label=f'{gan_name} Pruned Val')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def compare_generated_images(acgan, dcgan, wgan, began, num_images=10):
    noise = np.random.normal(0, 1, (num_images, acgan.latent_dim))
    labels = np.random.randint(0, acgan.num_classes, num_images)

    # Generate images from each GAN
    gen_imgs_acgan = acgan.generator.predict([noise, labels])
    gen_imgs_dcgan = dcgan.generator.predict(noise)
    gen_imgs_wgan = wgan.generator.predict(noise)
    gen_imgs_began = began.generator.predict(noise)

    # Plot the generated images
    fig, axes = plt.subplots(5, num_images, figsize=(15, 10))

    # ACGAN
    for i in range(num_images):
        axes[0, i].imshow(gen_imgs_acgan[i, :, :, 0], cmap='gray')
        axes[0, i].axis('off')

    # DCGAN
    for i in range(num_images):
        axes[1, i].imshow(gen_imgs_dcgan[i, :, :, 0], cmap='gray')
        axes[1, i].axis('off')

    # WGAN
    for i in range(num_images):
        axes[2, i].imshow(gen_imgs_wgan[i, :, :, 0], cmap='gray')
        axes[2, i].axis('off')

    # BEGAN
    for i in range(num_images):
        axes[3, i].imshow(gen_imgs_began[i, :, :, 0], cmap='gray')
        axes[3, i].axis('off')

    axes[0, 0].set_ylabel('ACGAN')
    axes[1, 0].set_ylabel('DCGAN')
    axes[2, 0].set_ylabel('WGAN')
    axes[3, 0].set_ylabel('BEGAN')

    plt.tight_layout()
    plt.show()


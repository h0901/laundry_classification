import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def compare_generated_images(acgan, dcgan, began, wgan, num_images=10):
    noise = np.random.normal(0, 1, (num_images, acgan.latent_dim))
    labels = np.random.randint(0, acgan.num_classes, num_images)

    # Generate images from each GAN
    gen_imgs_acgan = acgan.generator.predict([noise, labels])
    gen_imgs_dcgan = dcgan.generator.predict([noise, labels])
    gen_imgs_wgan = wgan.generator.predict([noise, labels])
    gen_imgs_began = began.generator.predict([noise, labels])

    #Plot the generated images
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

def plot_accuracy_heatmap(history_normal, history_augmented, history_pruned_augmented, gan_name):
    # Combine accuracy values into a 2D array
    data = np.array([
        history_normal.history['val_accuracy'],
        history_augmented.history['val_accuracy'],
        history_pruned_augmented.history['val_accuracy']
    ])

    labels = ['Normal', f'{gan_name}', f'{gan_name} Pruned']
    plt.figure(figsize=(12, 4))
    sns.heatmap(data, annot=False, cmap='viridis', cbar=True,
                xticklabels=5, yticklabels=labels)
    plt.title('Validation Accuracy Heatmap')
    plt.xlabel('Epoch')
    plt.ylabel('Model Type')
    plt.tight_layout()
    plt.show()

def compare_gan_heatmaps(gans: dict, num_images=5):
    # Get latent_dim and num_classes from first GAN
    sample_gan = list(gans.values())[0]
    latent_dim = sample_gan.latent_dim
    num_classes = getattr(sample_gan, 'num_classes', None)

    noise = np.random.normal(0, 1, (num_images, latent_dim))
    labels = np.random.randint(0, num_classes, num_images) if num_classes is not None else None

    fig, axes = plt.subplots(len(gans), num_images, figsize=(num_images * 2, len(gans) * 2.2))

    for row_idx, (gan_name, gan_model) in enumerate(gans.items()):
        if gan_name.lower() == 'acgan' and labels is not None:
            gen_imgs = gan_model.generator.predict([noise, labels])
        else:
            gen_imgs = gan_model.generator.predict([noise, labels])

        for col_idx in range(num_images):
            ax = axes[row_idx, col_idx] if len(gans) > 1 else axes[col_idx]
            sns.heatmap(gen_imgs[col_idx, :, :, 0], ax=ax, cmap='magma',
                        cbar=False, xticklabels=False, yticklabels=False)

            if col_idx == 0:
                ax.set_ylabel(gan_name, fontsize=12)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Generated Image Heatmaps from Different GANs", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

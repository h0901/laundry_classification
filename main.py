import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow_model_optimization as tfmot

from visualize import compare_generated_images, plot_full_history_comparison

import sys
sys.path.append("./gans")
from acgan import ACGAN
from dcgan import DCGAN
from wgan import WGAN
from began import BEGAN

sys.path.append("./models")
from cnn import CNN  

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

# Train original CNN
cnn = CNN(input_shape=(64, 64, 1))  # Grayscale images with 1 channel
history_cnn = cnn.train(x_train, y_train, x_test, y_test, epochs=10)
cnn.save_model()

# Function to train any GAN (ACGAN, DCGAN, WGAN, BEGAN, Pix2Pix)

# Initialize the GANs
acgan = ACGAN()
dcgan = DCGAN()
wgan = WGAN()
began = BEGAN()

os.makedirs('saved_models', exist_ok=True)

def train_acgan(gan, x_train, y_train, epochs=1000, batch_size=8):
        half_batch = batch_size // 2
        augmented_images = []
        augmented_labels = []
        
        for epoch in range(epochs):
            # Select random half batch of real images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            
            # Convert idx to tf.int32 tensor for TensorFlow compatibility
            idx = tf.convert_to_tensor(idx, dtype=tf.int32)
            
            # Use tf.gather to index x_train and y_train
            imgs = tf.gather(x_train, idx)
            labels = tf.gather(y_train, idx)
            
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

def train_dcgan(dcgan, x_train, y_train, epochs=1000, batch_size=8):
    half_batch = batch_size // 2
    augmented_images = []
    augmented_labels = []

    for epoch in range(epochs):
        # Select random half batch of real images
        idx = np.random.randint(0, x_train.shape[0], half_batch)

        # Convert idx to a TensorFlow tensor with dtype tf.int32
        idx = tf.convert_to_tensor(idx, dtype=tf.int32)

        # Use tf.gather to index x_train and y_train
        imgs, labels = tf.gather(x_train, idx), tf.gather(y_train, idx)

        # Generate random noise and labels for fake images
        noise = np.random.normal(0, 1, (half_batch, dcgan.latent_dim))
        gen_labels = np.random.randint(0, dcgan.num_classes, half_batch)
        gen_imgs = dcgan.generator.predict([noise, gen_labels])

        # Train the discriminator on real and fake images
        d_loss_real = dcgan.discriminator.train_on_batch([imgs, labels], np.ones((half_batch, 1)))
        d_loss_fake = dcgan.discriminator.train_on_batch([gen_imgs, gen_labels], np.zeros((half_batch, 1)))

        # Train the generator (via the combined model)
        noise = np.random.normal(0, 1, (batch_size, dcgan.latent_dim))
        sampled_labels = np.random.randint(0, dcgan.num_classes, batch_size)
        g_loss = dcgan.combined.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))

        # Store augmented data for later training the CNN
        augmented_images.extend(gen_imgs)
        augmented_labels.extend(gen_labels)

        # Print the losses at regular intervals
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {0.5 * np.add(d_loss_real, d_loss_fake)}, G Loss: {g_loss}")

    # Convert augmented data to numpy arrays after training is done
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels

def train_wgan(wgan, x_train, y_train, epochs=1000, batch_size=8, n_critic=5):
    half_batch = batch_size // 2
    augmented_images = []
    augmented_labels = []

    for epoch in range(epochs):
        # Train the critic for n_critic iterations
        for _ in range(n_critic):
            # Select random half batch of real images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            
            # Convert idx to TensorFlow-compatible tensor for indexing
            idx = tf.convert_to_tensor(idx, dtype=tf.int32)

            # Using tf.gather to index the images and labels
            imgs = tf.gather(x_train, idx)
            labels = tf.gather(y_train, idx)

            # Generate random noise and labels for fake images
            noise = np.random.normal(0, 1, (half_batch, wgan.latent_dim))
            gen_labels = np.random.randint(0, wgan.num_classes, half_batch)
            gen_imgs = wgan.generator.predict([noise, gen_labels])

            # Train the critic on real and fake images
            d_loss_real = wgan.critic.train_on_batch([imgs, labels], np.ones((half_batch, 1)))
            d_loss_fake = wgan.critic.train_on_batch([gen_imgs, gen_labels], -np.ones((half_batch, 1)))

        # Train the generator (via the combined model)
        noise = np.random.normal(0, 1, (batch_size, wgan.latent_dim))
        sampled_labels = np.random.randint(0, wgan.num_classes, batch_size)
        g_loss = wgan.combined.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))

        # Store augmented data for later training the CNN
        augmented_images.extend(gen_imgs)
        augmented_labels.extend(gen_labels)

        # Print the losses at regular intervals
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss_real + d_loss_fake}, G Loss: {g_loss}")

    # Convert augmented data to numpy arrays after training is done
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels

def train_began(began, x_train, y_train, epochs=1000, batch_size=8, n_critic=5):
    half_batch = batch_size // 2
    augmented_images = []
    augmented_labels = []

    for epoch in range(epochs):
        # Train the discriminator for n_critic iterations
        for _ in range(n_critic):
            # Select random half batch of real images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            
            # Convert idx to TensorFlow-compatible tensor for indexing
            idx = tf.convert_to_tensor(idx, dtype=tf.int32)

            # Using tf.gather to index the images and labels
            imgs = tf.gather(x_train, idx)
            labels = tf.gather(y_train, idx)

            # Generate random noise and labels for fake images
            noise = np.random.normal(0, 1, (half_batch, began.latent_dim))
            gen_labels = np.random.randint(0, began.num_classes, half_batch)
            gen_imgs = began.generator.predict([noise, gen_labels])

            # Train the discriminator on real and fake images
            d_loss_real = began.discriminator.train_on_batch([imgs, labels], np.ones((half_batch, 1)))
            d_loss_fake = began.discriminator.train_on_batch([gen_imgs, gen_labels], -np.ones((half_batch, 1)))

        # Train the generator (via the combined model)
        noise = np.random.normal(0, 1, (batch_size, began.latent_dim))
        sampled_labels = np.random.randint(0, began.num_classes, batch_size)
        g_loss = began.combined.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))

        # Store augmented data for later training the CNN
        augmented_images.extend(gen_imgs)
        augmented_labels.extend(gen_labels)

        # Print the losses at regular intervals
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss_real + d_loss_fake}, G Loss: {g_loss}")

    # Convert augmented data to numpy arrays after training is done
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels


def train_cnn_with_augmented_data(augmented_images, augmented_labels, x_test, y_test, gan_name):
    # Train CNN on augmented data
    cnn = CNN(input_shape=(64, 64, 1))  # Grayscale images with 1 channel
    history_cnn_augmented = cnn.train(augmented_images, augmented_labels, x_test, y_test, epochs=10)

    # Apply pruning to the trained CNN
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=np.ceil(len(augmented_images) / 32).astype(np.int32) * 5
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(cnn.model, **pruning_params)

    # Unfreeze last few layers for fine-tuning
    for layer in pruned_model.layers[-4:]:
        layer.trainable = True

    pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Pruning callback
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

    # Fine-tune the pruned model
    history_pruned_augmented = pruned_model.fit(
        augmented_images, augmented_labels,
        epochs=5,
        validation_data=(x_test, y_test),
        callbacks=[pruning_callback]
    )

    # Compare all three models
    plot_full_history_comparison(history_cnn, history_cnn_augmented, history_pruned_augmented, gan_name)

    print(f"Training history for {gan_name} with pruning plotted.")

    return history_cnn_augmented, history_pruned_augmented


def generate_images(generator, num_samples=10000):
    noise = np.random.normal(0, 1, (num_samples, 100))
    labels = np.random.randint(0, 10, (num_samples, 1))
    generated_images = generator.predict([noise, labels])
    return np.concatenate((x_train, generated_images)), np.concatenate((y_train, labels.squeeze()))

print('Starting ACGAN')
augmented_images_acgan, augmented_labels_acgan = train_acgan(acgan, x_train, y_train)

print('Starting DCGAN')
augmented_images_dcgan, augmented_labels_dcgan = train_dcgan(dcgan, x_train, y_train)

print('Starting WGAN')
augmented_images_wgan, augmented_labels_wgan = train_wgan(wgan, x_train, y_train)

print('Starting BEGAN')
augmented_images_began, augmented_labels_began = train_began(began, x_train, y_train)

# Normalize or clip all GAN outputs if needed
augmented_images_acgan = np.clip(augmented_images_acgan, 0, 1)
augmented_images_dcgan = np.clip(augmented_images_dcgan, 0, 1)
augmented_images_wgan  = np.clip(augmented_images_wgan,  0, 1)
augmented_images_began = np.clip(augmented_images_began, 0, 1)

history_acgan, history_acgan_pruned = train_cnn_with_augmented_data(
    np.concatenate((x_train, augmented_images_acgan), axis=0),
    np.concatenate((y_train, augmented_labels_acgan), axis=0),
    x_test, y_test, "ACGAN"
)

acgan.generator.save('saved_models/acgan_generator.h5')
acgan.discriminator.save('saved_models/acgan_discriminator.h5')
acgan.combined.save('saved_models/acgan_combined.h5')

history_dcgan, history_dcgan_pruned = train_cnn_with_augmented_data(
    np.concatenate((x_train, augmented_images_dcgan), axis=0),
    np.concatenate((y_train, augmented_labels_dcgan), axis=0),
    x_test, y_test, "DCGAN"
)

dcgan.generator.save('saved_models/dcgan_generator.h5')
dcgan.discriminator.save('saved_models/dcgan_discriminator.h5')
dcgan.combined.save('saved_models/dcgan_combined.h5')

history_wgan, history_wgan_pruned = train_cnn_with_augmented_data(
    np.concatenate((x_train, augmented_images_wgan), axis=0),
    np.concatenate((y_train, augmented_labels_wgan), axis=0),
    x_test, y_test, "WGAN"
)

wgan.generator.save('saved_models/wgan_generator.h5')
wgan.discriminator.save('saved_models/wgan_critic.h5')
wgan.combined.save('saved_models/wgan_combined.h5')

history_began, history_began_pruned = train_cnn_with_augmented_data(
    np.concatenate((x_train, augmented_images_began), axis=0),
    np.concatenate((y_train, augmented_labels_began), axis=0),
    x_test, y_test, "BEGAN"
)

began.generator.save('saved_models/began_generator.h5')
began.discriminator.save('saved_models/began_discriminator.h5')
began.combined.save('saved_models/began_combined.h5')

compare_generated_images(acgan, dcgan, wgan, began)

print("Models saved successfully!")

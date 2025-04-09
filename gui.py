import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from gans.acgan import ACGAN
from gans.dcgan import DCGAN
from models.cnn import CNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class LaundryGANApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laundry Tracker - GAN Augmentation & Classification")
        self.root.geometry("600x700")  # Increase height for better UI
        self.root.config(bg='white')  # Set background to white

        self.gan_options = {"ACGAN": ACGAN, "DCGAN": DCGAN}  # Add other GANs here
        self.selected_gan = tk.StringVar(value="ACGAN")

        cnn = CNN(input_shape=(64, 64, 1))

        # Load the pre-trained CNN model from main.py
        try:
            self.cnn_model = cnn.load_model(path='saved_models/cnn.h5')
            if self.cnn_model is None:
                raise ValueError("Model loading failed.")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Model Loading Error", f"Failed to load the CNN model. Please check the model path.")
            return  # Stop further execution if the model is not loaded correctly

        self.init_ui()

    def init_ui(self):
        tk.Label(self.root, text="Upload Laundry Image:", bg='white', font=("Arial", 14)).pack(pady=10)
        tk.Button(self.root, text="Choose File", command=self.upload_image, bg='pink', relief="flat", font=("Arial", 12)).pack(pady=5)

        self.image_label = tk.Label(self.root, bg='white')
        self.image_label.pack(pady=10)

        tk.Label(self.root, text="Select GAN for Augmentation:", bg='white', font=("Arial", 14)).pack(pady=10)
        tk.OptionMenu(self.root, self.selected_gan, *self.gan_options.keys()).pack(pady=5)

        tk.Button(self.root, text="Apply GAN Augmentation", command=self.apply_gan, bg='pink', relief="flat", font=("Arial", 12)).pack(pady=10)
        tk.Button(self.root, text="Classify with CNN", command=self.classify_image, bg='pink', relief="flat", font=("Arial", 12)).pack(pady=5)

        self.result_label = tk.Label(self.root, text="Result: ", font=("Arial", 12), bg='white')
        self.result_label.pack(pady=10)

        tk.Button(self.root, text="Visualize GAN Comparison", command=self.visualize_gans_comparison, bg='pink', relief="flat", font=("Arial", 12)).pack(pady=20)
        tk.Button(self.root, text="Visualize Loss & Accuracy Graphs", command=self.plot_training_graphs, bg='pink', relief="flat", font=("Arial", 12)).pack(pady=20)
        tk.Button(self.root, text="Visualize Image Histogram", command=self.plot_image_histogram, bg='pink', relief="flat", font=("Arial", 12)).pack(pady=20)

    def preprocess_image(self, image_path):
        """Resize and convert the uploaded image to shape (64, 64, 1)"""
        image = Image.open(image_path)
        
        # Resize the image to 64x64
        image = image.resize((64, 64))
        
        # Convert the image to grayscale (if it's RGB, it will be converted to single channel)
        image = image.convert('L')  # 'L' mode is for grayscale
        
        # Convert to NumPy array and reshape to (64, 64, 1)
        image_array = np.array(image)
        
        # Add a channel dimension to make it (64, 64, 1)
        image_array = image_array[..., np.newaxis]
        
        return image_array

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Preprocess the image (resize, grayscale, and reshape)
            self.image_array = self.preprocess_image(file_path)
            self.display_image()

    def display_image(self):
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.image_array.reshape(64, 64)))  # Convert back to Image
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        
    def apply_gan(self):
        if hasattr(self, 'image_array') and self.image_array is not None:
            gan_type = self.selected_gan.get()
            gan = self.gan_options[gan_type]()  # Instantiate the selected GAN
        
        # Normalize the image
            self.image_array = np.array(self.image_array) / 127.5 - 1  # Normalize the image
        
        # Apply GAN augmentation to the image_array
            augmented_img = gan.generate_samples(1, class_label=0)[0]  # Generate augmented image
        
        # Rescale the augmented image back to [0, 255]
            augmented_img_rescaled = (augmented_img + 1) * 127.5
            augmented_img_rescaled = np.clip(augmented_img_rescaled, 0, 255).astype(np.uint8)  # Ensure values are in valid range

        # Debugging: Check the shape and type of the augmented image
            print("Augmented image shape:", augmented_img_rescaled.shape)
            print("Augmented image dtype:", augmented_img_rescaled.dtype)

        # Convert to grayscale and remove the channel dimension
            augmented_img_rescaled = augmented_img_rescaled.reshape(64, 64)  # Ensure it's 2D for grayscale
        
        # Debugging: Check the shape and type after reshaping
            print("Reshaped augmented image shape:", augmented_img_rescaled.shape)
            print("Reshaped augmented image dtype:", augmented_img_rescaled.dtype)

        # Convert the augmented image into a format suitable for display
            self.image = Image.fromarray(augmented_img_rescaled)  # This should work now
        
            self.display_image()  # Display the augmented image
            messagebox.showinfo("GAN Augmentation", f"Applied {gan_type} augmentation")
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")

    def classify_image(self):
    # Check if an image is uploaded
        if hasattr(self, 'image_array') and self.image_array is not None:
        # Normalize and reshape the image
            img_array = self.image_array.reshape((1, 64, 64, 1)) / 255.0

        # Ensure CNN model is loaded
            if self.cnn_model is None:
                messagebox.showerror("Error", "CNN model not loaded.")
                return

        # Get prediction probabilities
            prediction = self.cnn_model.predict(img_array)
            predicted_class = np.argmax(prediction)

            # Print all prediction probabilities for debugging
            print("Prediction probabilities:", prediction[0])

            # Optionally: map class indices to Fashion MNIST labels
            class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

            label = class_labels[predicted_class]

            self.result_label.config(text=f"Predicted Class: {predicted_class} ({label})")
        else:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")


    def visualize_gans_comparison(self):
        gan_types = list(self.gan_options.keys())
        fig, axes = plt.subplots(1, len(gan_types) + 1, figsize=(15, 5))

        # Display the original image
        axes[0].imshow(self.image_array.reshape(64, 64), cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis('off')

        # Display images generated by each GAN
        for i, gan_type in enumerate(gan_types):
            gan = self.gan_options[gan_type]()
            augmented_img = gan.generate_samples(1, class_label=0)[0]  # Example class label
            axes[i + 1].imshow((augmented_img + 1) * 127.5, cmap='gray')  # Rescale to [0, 255]
            axes[i + 1].set_title(f"{gan_type}")
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_training_graphs(self):
        # Example loss and accuracy data (Replace with actual model history data)
        history = {
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
            "val_accuracy": [0.5, 0.65, 0.75, 0.8, 0.85],
            "loss": [0.6, 0.5, 0.4, 0.3, 0.2],
            "val_loss": [0.7, 0.6, 0.5, 0.45, 0.4]
        }

        # Plot accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_image_histogram(self):
        # Convert image to grayscale if needed
        gray_image = np.array(self.image_array.reshape(64, 64))

        plt.figure(figsize=(6, 6))
        plt.hist(gray_image.flatten(), bins=256, color='gray', alpha=0.7)
        plt.title('Pixel Intensity Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = LaundryGANApp(root)
    root.mainloop()

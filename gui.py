import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from gans.acgan import ACGAN
from gans.dcgan import DCGAN
from cnn import build_cnn, load_model

class LaundryGANApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laundry Tracker - GAN Augmentation & Classification")
        self.root.geometry("600x400")
        
        self.gan_options = {"ACGAN": ACGAN, "DCGAN": DCGAN}  # Add other GANs here
        self.selected_gan = tk.StringVar(value="ACGAN")
        
        self.cnn_model = load_model()
        self.init_ui()
    
    def init_ui(self):
        tk.Label(self.root, text="Upload Laundry Image:").pack()
        tk.Button(self.root, text="Choose File", command=self.upload_image).pack()
        
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        
        tk.Label(self.root, text="Select GAN for Augmentation:").pack()
        tk.OptionMenu(self.root, self.selected_gan, *self.gan_options.keys()).pack()
        
        tk.Button(self.root, text="Apply GAN Augmentation", command=self.apply_gan).pack()
        tk.Button(self.root, text="Classify with CNN", command=self.classify_image).pack()
        
        self.result_label = tk.Label(self.root, text="Result: ", font=("Arial", 12))
        self.result_label.pack()
    
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path).resize((64, 64))
            self.display_image()
    
    def display_image(self):
        img_tk = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
    
    def apply_gan(self):
        gan_type = self.selected_gan.get()
        gan = self.gan_options[gan_type]()
        self.image_array = np.array(self.image) / 127.5 - 1
        augmented_img = gan.generate_samples(1, class_label=0)[0]  # Example class label
        self.image = Image.fromarray(((augmented_img + 1) * 127.5).astype(np.uint8))
        self.display_image()
        messagebox.showinfo("GAN Augmentation", f"Applied {gan_type} augmentation")
    
    def classify_image(self):
        img_array = np.array(self.image).reshape((1, 64, 64, 3)) / 255.0
        prediction = np.argmax(self.cnn_model.predict(img_array))
        self.result_label.config(text=f"Result: {prediction}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LaundryGANApp(root)
    root.mainloop()

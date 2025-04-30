# Laundry Classification

This project classifies laundry images into different clothing categories using deep learning. It uses GANs (Generative Adversarial Networks) for data augmentation and includes a simple GUI for interactive classification.

## Features

- Image classification of clothing/laundry items
- GAN-based data augmentation with PyTorch
- Tkinter GUI for user-friendly classification
- Model evaluation and visualization scripts

## Directory Structure

```text
laundry_classification/
├── gans/              # GAN training scripts
├── gans_ipynb/        # GAN experiments in notebooks
├── models/            # Pretrained classifier models
├── saved_models/      # Model checkpoints
├── gui.py             # GUI for image classification
├── main.py            # Main script to run the classifier
├── evaluate_gans.py   # Evaluate GAN output quality
├── visualize.py       # Visualize results or dataset
├── loaded_models.py   # Model loading helper
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation 
```

Setup Instructions
1. Clone the Repository
git clone https://github.com/h0901/laundry_classification.git
cd laundry_classification

2. (Optional) Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

How to Run:-
Launch the GUI
python gui.py

Train GAN
python gans/train_gan.py

Evaluate GAN Outputs
python evaluate_gans.py

Visualize Data or Results
python visualize.py

Notes
Make sure your dataset is organized correctly before training.
Trained models and GAN-generated samples will be saved under saved_models/ or relevant subdirectories.
The GUI supports file upload to classify new laundry images using pretrained models.

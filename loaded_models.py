import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow_model_optimization as tfmot

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

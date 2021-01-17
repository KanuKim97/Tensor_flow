import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt

tfds.disable_progress_bar()
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

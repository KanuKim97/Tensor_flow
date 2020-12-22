from numpy.lib.npyio import load
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

class_names = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_img[0])
plt.colorbar()
plt.grid(False)
plt.show

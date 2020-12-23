from numpy.lib.npyio import load
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine import input_layer

fashion_mnist = keras.datasets.fashion_mnist

(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

class_names = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""
Labels
0 : Top
1 : Trouser
2 : Pullover
3 : Dress
4 : Coat
5 : Sandal
6 : Shirt
7 : Sneaker
8 : Bag
9 : Ankle Boot
"""

print(train_img.shape) # Train_set 
print(len(train_labels)) # Train_set_Label
print(train_labels) # Labels
print(test_img.shape) # Test_set

#Adjustment Size -> Range(0~1)
train_img = train_img / 255.0
test_img = test_img / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_img, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_img, test_labels, verbose=2)
print("\n test Accuracy: ", test_acc)
print("\n test_loss: ", test_loss)

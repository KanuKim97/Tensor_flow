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

#Predictions
predictions = model.predict(test_img)
#Reliability
predictions[0]

#Compare
np.argmax(predictions[0])
test_labels[0]

def plot_img(i, prediction_arr, true_label, img):
    prediction_arr, true_label, img = prediction_arr[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(prediction_arr)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(prediction_arr),
    class_names[true_label]), color = color)

def plot_value_arr(i, prediction_arr, true_label):
    prediction_arr, true_label = prediction_arr[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_arr, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_arr)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_img = num_rows*num_cols
plt.figure(figsize=(2*num_cols, 2*num_rows))
for i in range(num_img):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_img(i, predictions, test_labels, test_img)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_arr(i, predictions, test_labels)
plt.show()

img = test_img[0]
img = (np.expand_dims(img,0))

predictions_single = model.predict(img)

plot_value_arr(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation = 45)
np.argmax(predictions_single[0])
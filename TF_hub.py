from re import VERBOSE
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import binary_crossentropy 
import tensorflow_hub as hub
import tensorflow_datasets as tfds

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split = ('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

#print(train_examples_batch)
#print(train_labels_batch)

embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

hub_layer = hub.KerasLayer(
    embedding, input_shape = [],
    dtype =  tf.string, trainable=True
)

# print(hub_layer(train_examples_batch[:3]))

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(
    optimizer='adam',
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_data.shuffle(10000).batch(512),
    epochs = 30,
    validation_data = validation_data.batch(512),
    verbose=1
)

result = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, result):
  print("%s: %.3f" % (name, value))
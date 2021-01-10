import tensorflow as tf
import tensorflow_hub as tf_hub
from tensorflow import keras

import matplotlib.pylab as plt
import numpy as np 
import PIL.Image as Image

classiifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

IMAGE_SHAPE = (224,224)

classifier = tf.keras.Sequential([
    tf_hub.KerasLayer(classiifier_url, input_shape = IMAGE_SHAPE+(3,))
])

grace_hopper  = tf.keras.utils.get_file('image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape

result = classifier.predict(grace_hopper[np.newaxis, ...])
predicted_class = np.argmax(result[0], axis = -1)
labels_path = tf.keras.utils.get_file('','')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())

data_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar = True
)

image_generator =  tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

result_batch = classifier.predict(image_batch)
result_batch.shape
predicted_class_names = imagenet_labels[mp.argamx(result_batch, axis = -1)]
predicted_class_names

plt.figure(figsize=(10, 9 ))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6.5, n+1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_names[n])
    plt.axis('off')
_ = plt.subtitle("ImageNet Predictions")

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

feature_extractor_layer = tf_hub.KerasLayer(
    feature_extractor_url, 
    input_shape=(244,244,3)
)

feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    keras.layers.Dense(image_data.num_classes, activation='softmax')
])
model.summary()

predictions = model(image_batch)
predictions.shape

model.compile(
    optimizer= tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['acc']
)
import tensorflow as tf 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def plot_graphs(history, metrics):
    plt.plot(history.history[metrics])
    plt.plot(history.history['val_'+metrics], '')
    plt.xlabel("Epochs")
    plt.ylabel(metrics)
    plt.legend([metrics, 'val_'+metrics])
    plt.show()

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder
print('Vocabulary Size {}'.format(encoder.vocab_size))

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The Original String: {}'.format(original_string))

assert original_string == sample_string

for index in encoded_string:
    print('{} -----> {}'.format(index, encoder.decode([index])))

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_dataset.padded_batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs= 10,
    validation_data= test_dataset,
    validation_steps= 30
)

test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sample_pred_text, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
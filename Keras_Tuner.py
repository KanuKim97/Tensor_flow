import tensorflow as tf
from tensorflow import keras

import IPython
import kerastuner as kt
from traitlets.traitlets import validate

(img_Train, label_Train), (img_Test, label_Test) = keras.datasets.fashion_mnist.load_data()

img_Train = img_Train.astype('float32') / 255.0
img_Test = img_Test.astype('float32') / 255.0

def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    
    hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
    model.add(keras.layers.Dense(units= hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))
    
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

tuner = kt.Hyperband(
    model_builder, objective='val_accuracy', max_epochs=30,
    factor=3, directory='my_dir', project_name='intro_to_kt'
)

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_trian_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)

tuner.search(
    img_Train, label_Train, epochs = 10,
    validation_data = (img_Test, label_Test),
    callbacks = [ClearTrainingOutput()]
    )

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
model.fit(
    img_Train, label_Train, epochs = 10, 
    validation_data = (img_Test, label_Test)
)
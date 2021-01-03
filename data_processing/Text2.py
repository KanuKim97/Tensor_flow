import collections
import pathlib
import re
import string

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

BUFFER_SIZE = 50000
BATCH_SIZE = 64
VALIDATION_SIZE = 5000

Directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
File_Names = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in File_Names:
    text_dir = utils.get_file(name, origin=Directory_url + name)

parent_dir = pathlib.Path(text_dir).parent
list(parent_dir.iterdir())
print(list(parent_dir.iterdir()))

def labeler(example, index):
    return example, tf.cast(index, tf.int64)

labeled_data_sets = []

for i, file_name in enumerate(File_Names):
    lines_dataset = tf.data.TextLineDataset(str(parent_dir/file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False
)

for text, label in all_labeled_data.take(10):
    print("Sentence: ", text.numpy())
    print("Label: ", label.numpy())
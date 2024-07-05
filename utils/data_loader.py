import os
from functools import partial

import tensorflow as tf
import numpy as np
import pandas as pd


def load_data(cluster_size: int = 5000, num_cls: int = 8, clusters: int = 6):
    targets, data = [], []

    label_col = ['cat']
    # oh is short-hand from one-hot encoding
    oh_label_col = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    coord_col = ['min_x', 'max_y', 'max_x', 'min_y']
    
    labels = pd.read_csv('/workspace/data/labels.csv', index_col=[0])

    prefix = 0
    for idx in range(clusters):
        logits = np.load(os.path.join('/', 'workspace', 'data', 'numpy_array', f'imdb_1_{idx}.npy'))
        
        padding_size = cluster_size - logits.shape[0]
        st, end = prefix, prefix + logits.shape[0] - 1
        prefix += logits.shape[0]
        
        logits = _padding(logits, padding_size)
        corr_labels = _padding(labels.loc[st: end, label_col+oh_label_col+coord_col], padding_size)
        
        data.append(logits)
        targets.append(
            tf.convert_to_tensor(corr_labels,
                dtype=tf.float32
            )
        )

    targets = tf.reshape(tf.stack(targets, axis=0), shape=(-1, len(label_col)+len(oh_label_col)+len(coord_col)))
    data = tf.convert_to_tensor(np.stack(data, axis=0).reshape(-1, 224, 224, 3))

    return data, targets

def _padding(array: np.array, padding_size:int = 0):
    return np.concatenate([array, array[:padding_size]], axis=0)

def parse(proto, ft_desc):
    parsed_ft = tf.io.parse_single_example(proto, ft_desc)
    coord = parsed_ft['coord']
    label = parsed_ft['label']
    oh_label = parsed_ft['oh_label']
    image = parsed_ft['logit']

    return image, coord, label, oh_label

def load_data_tfrecord(path_to_tfrecord='/media/daniel/DatasetIMDB/imdb_chunks', 
                       class_num: int = 8):
    tfrecord_files = [os.path.join(path_to_tfrecord, f) for f in os.listdir(path_to_tfrecord) if f.endswith('.tfrecord')]

    feature_description = {
        'logit': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
        'oh_label': tf.io.FixedLenFeature([class_num], tf.float32),
        'coord': tf.io.FixedLenFeature([4], tf.float32) # min_x, max_x, min_y, max_y
    }
    parse_fn = partial(parse, ft_desc=feature_description)

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(parse_fn)
    
    return parsed_dataset
    
def to_dict_target(targets: tf.Tensor):
    dict_target = []

    for tgt in targets:
        t = {'labels': tf.gather(tgt, [0], axis=-1),
             'oh_labels': tf.gather(tgt, list(range(1, 9)), axis=-1),
             'boxes': tf.gather(tgt, [9, 10, 11, 12], axis=-1)}
        dict_target.append(t)
    
    return dict_target


if __name__ == "__main__":
    dataset = load_data_tfrecord('/media/daniel/DatasetIMDB/imdb_chunks')
    idx = 0
    for batch in dataset:
        img, coords, labels, oh_labels = batch
        print(tf.reshape(tf.cast(tf.io.decode_raw(img, tf.uint8), tf.float32), (-1, 224, 224, 3)))
        idx += 1
        if idx == 2:
            break

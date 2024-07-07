import tensorflow as tf


def padding_oh_labels(tensor: tf.Tensor):
    tf.where(tf.equal(tf.reduce_max(tensor), 0))

import tensorflow as tf


@tf.function
def cls_loss(tgt_labels, pred_cls):
    return tf.keras.losses.CategoricalCrossentropy()(tgt_labels, pred_cls)
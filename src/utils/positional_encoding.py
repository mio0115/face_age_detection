import math

import tensorflow as tf
import numpy as np


@tf.function
def gen_sineembed_for_position(pos_tensor, d_model):
    """
    Positional embedding of pos_tensor with depth d_model.

    Args:
        pos_tensor: The tensor to get positional embedding.
        d_model   : The depth of pos_tensor. (tf.shape(pos_tensor)[-1])

    Returns:
        tf.Tensor: A tensor which is offset of pos_tensor.
    """
    scale = tf.constant(2 * math.pi, dtype=tf.float16)
    fd_model = d_model // 2
    dim_t = tf.cast(tf.range(fd_model, dtype=tf.float32), dtype=tf.float16)
    dim_t = tf.constant(10000, dtype=tf.float16) ** tf.cast(
        tf.constant(2, dtype=tf.float16)
        * (dim_t // tf.constant(2, dtype=tf.float16))
        / fd_model,
        dtype=tf.float16,
    )

    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale

    pos_x = x_embed[..., tf.newaxis] / dim_t
    pos_y = y_embed[..., tf.newaxis] / dim_t

    pos_x = tf.concat([tf.sin(pos_x[..., 0::2]), tf.cos(pos_x[..., 1::2])], axis=-1)
    pos_y = tf.concat([tf.sin(pos_y[..., 0::2]), tf.cos(pos_y[..., 1::2])], axis=-1)
    pos = tf.concat([pos_x, pos_y], axis=2)
    return pos

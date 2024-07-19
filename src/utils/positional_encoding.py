import math

import tensorflow as tf


def gen_sineembed_for_position(pos_tensor, d_model: int = 512):
    """
    Positional embedding of pos_tensor with depth d_model.

    Args:
        pos_tensor: The tensor to get positional embedding.
        d_model   : The depth of pos_tensor. (tf.shape(pos_tensor)[-1])

    Returns:
        tf.Tensor: A tensor which is offset of pos_tensor.
    """
    scale = 2 * math.pi
    fd_model = d_model // 2
    dim_t = tf.range(fd_model, dtype=tf.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / fd_model)

    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale

    pos_x = x_embed[..., tf.newaxis] / dim_t
    pos_y = y_embed[..., tf.newaxis] / dim_t

    pos_x = tf.concat([tf.sin(pos_x[..., 0::2]), tf.cos(pos_x[..., 1::2])], axis=-1)
    pos_y = tf.concat([tf.sin(pos_y[..., 0::2]), tf.cos(pos_y[..., 1::2])], axis=-1)
    pos = tf.concat([pos_x, pos_y], axis=2)
    return pos


def with_position_embedding(pos_tensor, d_model: int = 512):
    return pos_tensor + gen_sineembed_for_position(pos_tensor, d_model)

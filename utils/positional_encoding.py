import math

import tensorflow as tf
import numpy as np


def positional_encoding_sin_2d(height, width, channels):
    def get_positional_encoding(d_model, length):
        angles = np.array(
            [
                pos / pow(10000, 2 * (k // 2) / d_model)
                for pos in range(length)
                for k in range(d_model)
            ]
        )
        angles = angles.reshape(length, d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        return angles

    height_encoding = get_positional_encoding(d_model=channels // 2, length=height)
    width_encoding = get_positional_encoding(d_model=channels // 2, length=width)

    pos_encoding = np.zeros((height, width, channels))

    pos_encoding[..., : channels//2] = height_encoding[:, np.newaxis, :]
    pos_encoding[..., channels//2 :] = width_encoding[np.newaxis, ...]

    return tf.constant(pos_encoding, dtype=tf.float32)

def gen_sineembed_for_position(pos_tensor, d_model):
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
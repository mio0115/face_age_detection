import tensorflow as tf
from tensorflow.keras.layers import Dense  # type: ignore

from ...utils.positional_encoding import gen_sineembed_for_position


class SelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        heads_num,
        input_shape,
        output_shape,
        d_k,
        d_v,
        *args,
        **kwargs,
    ):
        super(SelfAttention, self).__init__(*args, **kwargs)

        self._heads_num = heads_num
        # shape = (sequence_len, embedding_dim)
        self._input_shape = input_shape
        # shape = (sequence_len, embedding_dim)
        self._output_shape = output_shape
        self._d_k = d_k
        self._d_v = d_v

        self._proj_to_query = Dense(units=self._d_k, use_bias=False)
        self._proj_to_key = Dense(units=self._d_k, use_bias=False)
        self._proj_to_value = Dense(units=self._d_v, use_bias=False)
        self._proj_to_output = Dense(units=output_shape[-1], use_bias=False)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def input_height(self):
        return self._input_shape[0]

    @property
    def input_width(self):
        return self._input_shape[1]

    @property
    def sequence_length(self):
        return self._input_shape[0]

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def heads_num(self):
        return self._heads_num

    @property
    def per_head_dim(self):
        return self.input_shape[-1] // self.heads_num

    def _split_heads(self, x):
        batch_size = tf.shape(x)[0]

        x = tf.reshape(
            x, (batch_size, self.sequence_length, self.heads_num, self.per_head_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, obj_pos_encoding: tf.Tensor | None = None):
        # shape should be (batch_size, sequence_length, embedding_dim)
        batch_size = tf.shape(inputs)[0]

        if obj_pos_encoding is not None:
            x_to_query_key = inputs + obj_pos_encoding
        else:
            x_to_query_key = inputs + gen_sineembed_for_position(
                inputs, d_model=self.input_shape[-1]
            )
        x_to_value = inputs

        # reshape inputs into (batch_size, heads_num, seq_len, head_embedding_dim)
        x_to_query_key = self._split_heads(x_to_query_key)
        x_to_value = self._split_heads(x_to_value)

        # compute Q, K and V
        # shape = (batch_size, heads_num, seq_len, d_k)
        q = self._proj_to_query(x_to_query_key)
        # shape = (batch_size, heads_num, seq_len, d_k)
        k = self._proj_to_key(x_to_query_key)
        # shape = (batch_size, heads_num, seq_len, d_v)
        v = self._proj_to_value(x_to_value)

        # shape = (batch_size, heads_num, seq_len, seq_len)
        a_sc = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) / tf.sqrt(
            tf.cast(self._d_k, tf.float32)
        )
        a_sc = tf.keras.activations.softmax(a_sc)
        output = tf.matmul(a_sc, v)

        # we need to then concatenate the output of all the result from matmul(a_sc, v)
        # the shape would be (batch_size, seq_len, output_embedding_dim)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(
            output, shape=(batch_size, self.sequence_length, self.heads_num * self._d_v)
        )
        output = self._proj_to_output(output)

        return output

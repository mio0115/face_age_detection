import tensorflow as tf
from tensorflow.keras.layers import Dense


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, obj_input_shape, heads_num, d_k, d_v):
        super(CrossAttention, self).__init__()

        # the input should be rank 2 (seq_len, embedding_dim)
        self._input_shape = obj_input_shape  # 256
        self._heads_num = heads_num
        self._d_k = d_k  # 512
        self._d_v = d_v  # 512

        self._proj_to_query = Dense(units=self._d_k, use_bias=False)
        self._proj_to_key = Dense(units=self._d_k, use_bias=False)
        self._proj_to_value = Dense(units=self._d_v, use_bias=False)
        self._proj_to_output = Dense(units=self.embedding_dim, use_bias=False)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def heads_num(self):
        return self._heads_num

    @property
    def sequence_length(self):
        return self._input_shape[0]

    @property
    def embedding_dim(self):
        return self._input_shape[-1]

    @property
    def per_head_dim(self):
        return self._input_shape[-1] // self._heads_num  # 256 // 8 = 32

    def _split_heads(self, x, seq_len: int):
        batch_size = tf.shape(x)[0]

        # print(batch_size, self.sequence_length, self.heads_num, self.per_head_dim)
        # print(tf.shape(x))
        x = tf.ensure_shape(x, shape=(None, seq_len) + (self.input_shape[-1],))
        x = tf.reshape(
            x, shape=(batch_size, seq_len, self.heads_num, self.per_head_dim)
        )

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, obj_queries, query_obj_pos, encoder_output, key_encoder_pos):
        batch_size = tf.shape(obj_queries)[0]

        obj_queries = tf.ensure_shape(
            obj_queries,
            shape=(None,) + self.input_shape,
            name="check_sc_attn_input_shape",
        )

        # query
        obj_queries = self._split_heads(obj_queries, 5)
        query = tf.concat(
            [self._proj_to_query(obj_queries), self._split_heads(query_obj_pos, 5)],
            axis=-1,
        )  # to 512 (8, 32) -> (8, 64)

        # key
        encoder_output = self._split_heads(encoder_output, 49)
        key = tf.concat(
            [self._proj_to_key(encoder_output), self._split_heads(key_encoder_pos, 49)],
            axis=-1,
        )  # ()

        # value
        value = self._proj_to_value(encoder_output)

        a_cross = tf.nn.softmax(
            tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2]))
            / tf.sqrt(tf.cast(self.embedding_dim, dtype=tf.float32)),
            axis=-1,
        )
        o = tf.matmul(a_cross, value)  # (8, 512)
        o = tf.transpose(o, perm=[0, 2, 1, 3])
        o = tf.cast(o, dtype=tf.float32)
        o = tf.reshape(
            o, shape=(batch_size, self.sequence_length, self.heads_num * self._d_v)
        )  # 8 * 512 = 4096

        return self._proj_to_output(tf.cast(o, dtype=tf.float32))  # to 256

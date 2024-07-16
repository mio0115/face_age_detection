import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout  # type: ignore

from ..attention.self_attention import SelfAttention
from ..attention.pair_self_attention import PairSelfAttention
from ..attention.split_cross_attention import CrossAttention
from ...utils.positional_encoding import gen_sineembed_for_position


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        object_queries_shape,
        lambda_: float = 0.5,
        hidden_dim: int = 512,
        **kwargs
    ):
        super(DecoderBlock, self).__init__(**kwargs)

        self._lambda = lambda_
        self._self_attention = SelfAttention(
            heads_num=8,
            input_shape=object_queries_shape,
            output_shape=object_queries_shape,
            d_k=hidden_dim,
            d_v=hidden_dim,
        )
        self._pair_self_attention = PairSelfAttention(
            heads_num=8,
            input_shape=object_queries_shape,
        )
        self._cls_branch = ClsRegBranch(
            object_queries_shape=(object_queries_shape[0], hidden_dim // 2)
        )
        self._reg_branch = ClsRegBranch(
            object_queries_shape=(object_queries_shape[0], hidden_dim // 2)
        )
        self._proj_to_query = Dense(units=hidden_dim // 2, use_bias=False)
        self._proj_to_key = Dense(units=hidden_dim // 2, use_bias=False)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

        # rank is 2 (sequence length, embedding dim)
        self._object_queries_shape = object_queries_shape

        self._channels = hidden_dim // 2

    @property
    def object_queries_shape(self):
        return self._object_queries_shape

    @property
    def object_queries_seq_len(self):
        return self.object_queries_shape[0]

    @property
    def object_queries_embedding_dim(self):
        return self.object_queries_shape[1]

    def call(
        self,
        object_queries: tf.Tensor,
        encoder_output: tf.Tensor,
        obj_centers: tf.Tensor,
        obj_pos_encoding: tf.Tensor,
    ):
        batch_size = tf.shape(object_queries)[0]
        object_queries = tf.ensure_shape(
            object_queries,
            shape=(None,) + self.object_queries_shape,
            name="check_decoder_input",
        )

        obj_pos_encoding = gen_sineembed_for_position(
            pos_tensor=obj_centers, d_model=512
        )
        enc_pos_encoding = gen_sineembed_for_position(encoder_output, d_model=512)

        query_obj_pos = self._proj_to_query(obj_pos_encoding)
        key_pos = self._proj_to_key(enc_pos_encoding)

        query_obj_pos = tf.ensure_shape(
            query_obj_pos, shape=(None, self.object_queries_seq_len, 256)
        )

        o1 = self._self_attention(object_queries, obj_pos_encoding)
        o2 = self._pair_self_attention(object_queries, obj_pos_encoding, obj_centers)
        o = self._lambda * self.layer_norm1(object_queries + o1) + (
            1 - self._lambda
        ) * self.layer_norm2(object_queries + o2)
        o = tf.ensure_shape(
            o,
            shape=(
                None,
                self.object_queries_seq_len,
                self.object_queries_embedding_dim,
            ),
        )
        o_cls, o_reg = tf.split(o, num_or_size_splits=2, axis=-1)

        cls_output = self._cls_branch(o_cls, query_obj_pos, encoder_output, key_pos)
        reg_output = self._reg_branch(o_reg, query_obj_pos, encoder_output, key_pos)
        o = tf.ensure_shape(
            tf.concat([cls_output, reg_output], axis=-1),
            shape=(
                None,
                self.object_queries_seq_len,
                self.object_queries_embedding_dim,
            ),
        )
        return o


class ClsRegBranch(tf.keras.layers.Layer):
    def __init__(self, object_queries_shape: tuple[int], hidden_dim: int = 256):
        super(ClsRegBranch, self).__init__()

        self.cross_attn = CrossAttention(
            heads_num=8,
            d_k=hidden_dim,
            d_v=hidden_dim,
            obj_input_shape=object_queries_shape,
        )

        self.dense1 = Dense(units=hidden_dim, activation="relu")
        self.dense2 = Dense(units=hidden_dim)
        self.dropout = Dropout(0.1)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

    def call(self, inputs, query_obj_pos, encoder_output, key_pos):
        ca = self.cross_attn(inputs, query_obj_pos, encoder_output, key_pos)

        x = inputs + self.dropout(ca)
        x = self.layer_norm1(x)
        x_2 = self.dense2(self.dropout(self.dense1(x)))
        x = x + self.dropout(x_2)
        x = self.layer_norm2(x)

        return x

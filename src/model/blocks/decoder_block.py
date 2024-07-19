import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout  # type: ignore

from ..attention.self_attention import SelfAttentionV2
from ..attention.pair_self_attention import PairSelfAttentionV2
from ..attention.split_cross_attention import CrossAttention
from ...utils.positional_encoding import gen_sineembed_for_position


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        object_queries_shape,
        lambda_: float = 0.5,
        hidden_dim: int = 512,
        **kwargs,
    ):
        super(DecoderBlock, self).__init__(**kwargs)

        self._lambda = lambda_
        self._self_attn_layer = SelfAttentionV2(
            heads_num=8,
            input_shape=object_queries_shape,
            output_shape=object_queries_shape,
        )
        self._pair_attn_layer = PairSelfAttentionV2(
            heads_num=8,
            input_shape=object_queries_shape,
        )
        self._cls_branch = ClsRegBranchV2(
            attn_input_shape=object_queries_shape,
            attn_output_shape=(object_queries_shape[0], object_queries_shape[1] // 2),
            hidden_dim=hidden_dim // 2,
        )
        self._reg_branch = ClsRegBranchV2(
            attn_input_shape=object_queries_shape,
            attn_output_shape=(object_queries_shape[0], object_queries_shape[1] // 2),
            hidden_dim=hidden_dim // 2,
        )

        self._sa_proj_to_q_obj = Dense(units=hidden_dim, use_bias=False)
        self._sa_proj_to_q_pos = Dense(units=hidden_dim // 2, use_bias=False)
        self._sa_proj_to_k_obj = Dense(units=hidden_dim, use_bias=False)
        self._sa_proj_to_k_pos = Dense(units=hidden_dim // 2, use_bias=False)
        self._sa_proj_to_v_obj = Dense(units=hidden_dim, use_bias=False)

        self._ca_proj_to_q_obj = Dense(units=hidden_dim, use_bias=False)
        self._ca_proj_to_q_pos = Dense(units=hidden_dim // 2, use_bias=False)
        self._ca_proj_to_k_enc = Dense(units=hidden_dim // 2, use_bias=False)
        self._ca_proj_to_k_pos = Dense(units=hidden_dim // 2, use_bias=False)
        self._ca_proj_to_v_enc = Dense(units=hidden_dim, use_bias=False)

        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        self.dropout = Dropout(0.1)

        # rank is 2 (sequence length, embedding dim)
        self._object_queries_shape = object_queries_shape
        self._heads_num = 8

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

    def _split_heads(self, tensor):
        batch_size = tf.shape(tensor)[0]
        seq_len = tf.shape(tensor)[1]
        d_model = tf.shape(tensor)[2]

        tensor = tf.reshape(
            tensor,
            shape=(batch_size, seq_len, self._heads_num, d_model // self._heads_num),
        )
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def _combine_heads(self, tensor):
        batch_size = tf.shape(tensor)[0]
        seq_len = tf.shape(tensor)[2]
        d_model = tf.shape(tensor)[3]

        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        tensor = tf.reshape(
            tensor, shape=(batch_size, seq_len, self._heads_num * d_model)
        )
        return tensor

    def call(
        self,
        object_queries: tf.Tensor,
        encoder_output: tf.Tensor,
        obj_coords: tf.Tensor,
        obj_pos_encoding: tf.Tensor,
        obj_sin_embed: tf.Tensor,
    ):
        batch_size = tf.shape(object_queries)[0]
        object_queries = tf.ensure_shape(
            object_queries,
            shape=(None,) + self.object_queries_shape,
            name="check_decoder_input",
        )

        enc_pos_encoding = gen_sineembed_for_position(encoder_output, d_model=512)

        q_obj = self._sa_proj_to_q_obj(object_queries)
        q_pos = self._sa_proj_to_q_pos(obj_pos_encoding)
        # note that the object_queries is concated by cls_output and reg_output
        q_pos = tf.concat([q_pos, q_pos], axis=-1)

        k_obj = self._sa_proj_to_k_obj(object_queries)
        k_pos = self._sa_proj_to_k_pos(obj_pos_encoding)
        k_pos = tf.concat([k_pos, k_pos], axis=-1)

        v = self._split_heads(self._sa_proj_to_v_obj(object_queries))
        q = self._split_heads(q_obj + q_pos)
        k = self._split_heads(k_obj + k_pos)

        o1 = self._self_attn_layer(query=q, key=k, value=v)
        o2 = self._pair_attn_layer(query=q, key=k, value=v, top_k_centers=obj_coords)

        o = self._lambda * self.layer_norm1(object_queries + self.dropout(o1)) + (
            1 - self._lambda
        ) * self.layer_norm2(object_queries + self.dropout(o2))
        o = tf.ensure_shape(
            o,
            shape=(
                None,
                self.object_queries_seq_len,
                self.object_queries_embedding_dim,
            ),
        )
        o_cls, o_reg = tf.split(o, num_or_size_splits=2, axis=-1)

        q_obj = self._ca_proj_to_q_obj(o)
        q_pos = self._ca_proj_to_q_pos(obj_sin_embed)
        k_enc = self._ca_proj_to_k_enc(encoder_output)
        k_pos = self._ca_proj_to_k_pos(enc_pos_encoding)
        v2 = self._ca_proj_to_v_enc(encoder_output)
        v2 = tf.ensure_shape(v2, shape=(None, 49, 512))

        q_cls, q_reg = tf.split(q_obj, num_or_size_splits=2, axis=-1)
        q_cls = self._split_heads(q_cls)
        q_reg = self._split_heads(q_reg)
        q_pos = self._split_heads(q_pos)
        q_cls = tf.concat([q_cls, q_pos], axis=-1)
        q_reg = tf.concat([q_reg, q_pos], axis=-1)
        q_cls = self._combine_heads(q_cls)
        q_reg = self._combine_heads(q_reg)
        # q_cls = tf.ensure_shape(q_cls, shape=(None,) + self.object_queries_shape)
        # q_reg = tf.ensure_shape(q_reg, shape=(None,) + self.object_queries_shape)

        k_enc = self._split_heads(k_enc)
        k_pos = self._split_heads(k_pos)
        k = tf.concat([k_enc, k_pos], axis=-1)
        k = self._combine_heads(k)

        cls_output = self._cls_branch(inputs=o_cls, query=q_cls, key=k, value=v2)
        reg_output = self._reg_branch(inputs=o_reg, query=q_reg, key=k, value=v2)
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


class ClsRegBranchV2(tf.keras.layers.Layer):
    def __init__(self, attn_input_shape, attn_output_shape, hidden_dim: int = 256):
        super(ClsRegBranchV2, self).__init__()

        self.cross_attn_layer = SelfAttentionV2(
            heads_num=1, input_shape=attn_input_shape, output_shape=attn_output_shape
        )

        self.dense1 = Dense(units=hidden_dim * 4, activation="relu")
        self.dense2 = Dense(units=hidden_dim)
        self.dropout = Dropout(0.1)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

    def call(self, inputs, query, key, value):
        # tf.print(f"shape of query: {tf.shape(query)}")
        # tf.print(f"shape of key  : {tf.shape(key)}")
        # tf.print(f"shape of value: {tf.shape(value)}")
        ca = self.cross_attn_layer(
            query=query[tf.newaxis, ...],
            key=key[tf.newaxis, ...],
            value=value[tf.newaxis, ...],
        )
        # ca = tf.squeeze(ca, axis=0)

        x = inputs + self.dropout(ca)
        x = self.layer_norm1(x)
        x_2 = self.dense2(self.dropout(self.dense1(x)))
        x = x + self.dropout(x_2)
        x = self.layer_norm2(x)

        return x

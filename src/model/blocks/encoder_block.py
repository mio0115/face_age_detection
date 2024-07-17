import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization  # type: ignore

from ..attention.self_attention import SelfAttention, SelfAttentionV2
from ...utils.positional_encoding import gen_sineembed_for_position


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_shape,
        heads_num,
        d_k,
        d_v,
        block_idx,
        *args,
        **kwargs,
    ):
        super(EncoderBlock, self).__init__(*args, **kwargs)

        assert input_shape[-1] % heads_num == 0

        # self.self_attention_layer = SelfAttention(
        #    heads_num=heads_num,
        #   input_shape=input_shape,
        #  output_shape=input_shape,
        # d_k=d_k,
        # d_v=d_v,
        # name=f"encoder_block_{block_idx}_multi-head_self-attention",
        # )
        self.self_attn_layer = SelfAttentionV2(
            heads_num=heads_num,
            input_shape=input_shape,
            output_shape=input_shape,
            name=f"encoder_block_{block_idx}_multi-head_self-attention",
        )
        self.dense1 = Dense(
            units=2048,
            activation="relu",
            name=f"encoder_block_{block_idx}_dense_1",
        )
        self.dense2 = Dense(
            units=512,
            activation="linear",
            name=f"encoder_block_{block_idx}_dense_2",
        )
        self.dropout = Dropout(0.1)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

        self._heads_num = heads_num
        self._d_k = d_k
        self._d_v = d_v
        self._proj_to_q = Dense(units=d_k, use_bias=False)
        self._proj_to_k = Dense(units=d_k, use_bias=False)
        self._proj_to_v = Dense(units=d_v, use_bias=False)

        # shape = (height, width, embedding_dim)
        self._input_shape = input_shape

    @property
    def input_shape(self):
        return self._input_shape

    @tf.function
    def _split_heads(self, tensor, batch_size):
        tensor = tf.reshape(
            tensor,
            shape=(
                batch_size,
                self.input_shape[0],
                self._heads_num,
                self.input_shape[1] // self._heads_num,
            ),
        )
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = inputs

        to_q_k = x + gen_sineembed_for_position(x)
        to_q_k = self._split_heads(to_q_k, batch_size)
        q = self._proj_to_q(to_q_k)
        k = self._proj_to_k(to_q_k)

        to_v = self._split_heads(x, batch_size)
        v = self._proj_to_v(to_v)

        residual_x = self.self_attn_layer(query=q, key=k, value=v)
        x = x + self.dropout(residual_x)

        x = self.layer_norm1(x)
        residual_x = self.dense1(x)
        residual_x = self.dropout(residual_x)
        residual_x = self.dense2(residual_x)
        x = x + self.dropout(residual_x)
        x = self.layer_norm2(x)

        return x


if __name__ == "__main__":
    eb = EncoderBlock(
        input_shape=(7, 7, 512),
        heads_num=8,
        d_k=256,
        d_v=512,
        block_idx=1,
        positional_encoding=tf.random.normal(shape=(7, 7, 512)),
    )
    print(eb(tf.random.uniform(shape=(2, 7, 7, 512), maxval=255.0)))

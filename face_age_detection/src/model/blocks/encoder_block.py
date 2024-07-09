import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from ..attention.self_attention import SelfAttention


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

        self.self_attention_layer = SelfAttention(
            heads_num=heads_num,
            input_shape=input_shape,
            output_shape=input_shape,
            d_k=d_k,
            d_v=d_v,
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
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

        # shape = (height, width, embedding_dim)
        self._input_shape = input_shape

    @property
    def input_shape(self):
        return self._input_shape

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = inputs

        tmp_x = x
        x = self.self_attention_layer(x)
        x = self.dropout1(x)
        x = self.layer_norm1(x + tmp_x)

        tmp_x = x
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.layer_norm2(x + tmp_x)

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

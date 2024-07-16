import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Conv2D, Dense
from tensorflow.keras.models import Sequential

from .blocks.encoder_block import EncoderBlock
from .blocks.mini_detector import MiniDetector
from .blocks.decoder_block import DecoderBlock
from ..utils.positional_encoding import gen_sineembed_for_position


"""use IMDB-wiki https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ as training dataset
This model is an implementation of Object Detection with Split Transformer.

Reference:
    He, L., & Todorovic, S. (2022). Destr: Object detection with split transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9377-9386).
"""


class ObjDetSplitTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_shape,
        num_cls: int,
        hidden_dim: int = 512,
        num_encoder_blocks: int = 1,
        num_decoder_blocks: int = 6,
        top_k: int = 5,
    ):
        super(ObjDetSplitTransformer, self).__init__()

        # Shape of input and output are (224, 224, 3), (7, 7, 2048), repectively.
        # Used to extract features from the input.
        self._resnet_backbone = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )

        self._hidden_dim = hidden_dim
        # Class feed-forward network is to classify predicted objects.
        self._cls_ffn = tf.keras.layers.Dense(units=num_cls, activation="softmax")
        # Regression feed-froward network is to regress boundary boxes of each predicted objects
        self._reg_ffn = Sequential(
            [Dense(units=256), Dense(units=256), Dense(units=4)],
            name="regression_head_coord",
        )
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

        # Reduce channels from 2048 to 512
        self._reduce_dim = Conv2D(filters=hidden_dim, kernel_size=(1, 1))
        # Mini detector used to output object proposals to following decoders
        self._mini_detector = MiniDetector(
            input_shape=(7, 7, hidden_dim),
            cls_num=num_cls,
            top_k=top_k,
            reg_ffn=self._reg_ffn,
            name="mini-detector",
        )
        self._num_encoders = num_encoder_blocks
        self._num_decoders = num_decoder_blocks

        # Normal encoder in Transformer.
        self._encoder_blocks = Sequential(
            [
                EncoderBlock(
                    input_shape=(49, hidden_dim),
                    heads_num=8,
                    d_k=128,
                    d_v=256,
                    block_idx=idx,
                )
                for idx in range(num_encoder_blocks)
            ],
            name="EncoderBlocks",
        )
        # Modified deocoder according to DESTR paper.
        self._decoder_blocks = [
            DecoderBlock(
                object_queries_shape=(top_k, hidden_dim),
                lambda_=0.5,
                name=f"DecoderBlock_{idx}",
            )
            for idx in range(num_decoder_blocks)
        ]

    def call(self, inputs):
        # extend for the batch dimension
        inputs = tf.cond(
            tf.equal(tf.rank(inputs), 3),
            lambda: tf.expand_dims(inputs, axis=0),
            lambda: inputs,
        )

        batch_size = tf.shape(inputs)[0]

        inputs = tf.ensure_shape(inputs, [None, *self._resnet_backbone.input.shape[1:]])
        inputs = tf.keras.applications.resnet.preprocess_input(inputs)
        x = self._resnet_backbone(inputs)
        tf.debugging.check_numerics(x, "NaN detected from Backbone.")

        x = self._reduce_dim(x)  # reduce embedding dimension

        x = tf.reshape(x, shape=(batch_size, 49, self._hidden_dim))
        # generate positional encoding and add to x
        x += gen_sineembed_for_position(x, d_model=self._hidden_dim)

        x = self._encoder_blocks(x)
        encoder_output = x

        tf.debugging.check_numerics(x, "NaN detected from Encoder.")

        # reshape to (7, 7, -1) since there are conv layers in the mini-detector
        x = tf.reshape(x, shape=(batch_size, 7, 7, self._hidden_dim))
        # First of all, we take only the top k proposals from the mini-detector
        # all_proposals is used to train the mini-detector
        # the other three is for the following forwarding
        top_k_proposals, top_k_centers, top_k_pos, all_proposals = self._mini_detector(
            x
        )

        # top_k_proposals: (..., 512), first 256 for cls, last 256 for reg
        top_k_proposals = tf.stop_gradient(top_k_proposals)
        # top_k_centers: (batch_size, top_k, 4), coord of the center points of the k proposals
        top_k_centers = tf.stop_gradient(top_k_centers)
        # top_k_pos: (batch_size, top_k, 512), position embedding for the k proposals
        top_k_pos = tf.stop_gradient(top_k_pos)

        x = top_k_proposals
        tf.debugging.check_numerics(x, "NaN detected from MiniDetector.")

        for idx in range(self._num_decoders):
            x = self._decoder_blocks[idx](x, encoder_output, top_k_centers, top_k_pos)

        cls_x, reg_x = tf.split(x, num_or_size_splits=2, axis=-1)
        cls_x = self.layer_norm1(cls_x)
        reg_x = self.layer_norm2(reg_x)

        cls_output = self._cls_ffn(cls_x)
        bbox_output = self._reg_ffn(reg_x)

        # for top_k objects, we predict they would be the class with their maximum probability
        idx = tf.argmax(
            tf.reduce_max(cls_output, axis=-1), axis=-1, output_type=tf.int32
        )
        cls_output = tf.gather(cls_output, idx, batch_dims=1)
        bbox_output = tf.gather(bbox_output, idx, batch_dims=1)

        return cls_output, bbox_output, all_proposals


def build_model(
    input_shape=(224, 224, 3),
    num_cls=8,
    num_encoder_blocks=1,
    num_decoder_blocks=6,
    top_k=5,
):
    destr_block = ObjDetSplitTransformer(
        input_shape=input_shape,
        num_cls=num_cls,
        num_encoder_blocks=num_encoder_blocks,
        num_decoder_blocks=num_decoder_blocks,
        top_k=top_k,
    )

    img = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    cls_output, reg_output, total_proposals = destr_block(img)

    model = tf.keras.Model(
        inputs=img, outputs=[cls_output, reg_output, total_proposals]
    )

    return model


if __name__ == "__main__":
    model = ObjDetSplitTransformer(input_shape=(224, 224, 3), num_cls=8)

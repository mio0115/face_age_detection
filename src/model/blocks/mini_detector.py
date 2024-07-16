import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.activations import sigmoid  # type: ignore

from ...utils.positional_encoding import gen_sineembed_for_position


class MiniDetector(tf.keras.layers.Layer):
    def __init__(
        self, input_shape, cls_num: int, top_k: int, hidden_dim: int = 256, **kwargs
    ):
        super(MiniDetector, self).__init__(**kwargs)

        # rank of input_shape = 3
        # shape = (height, width, embedding_dim)
        # embedding_dim should be 128?
        self._input_shape = input_shape
        self._top_K = top_k
        self._hidden_dim = hidden_dim

        self._cls_conv = Sequential(
            [
                Conv2D(
                    filters=hidden_dim, kernel_size=(3, 3), strides=1, padding="same"
                )
                for _ in range(4)
            ],
            name="mini_detector_classify_conv",
        )
        self._reg_conv = Sequential(
            [
                Conv2D(
                    filters=hidden_dim, kernel_size=(3, 3), strides=1, padding="same"
                )
                for _ in range(4)
            ],
            name="mini_detector_regression_conv",
        )
        self._pos_conv = Sequential(
            [
                Conv2D(
                    filters=hidden_dim,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    activation="relu",
                )
                for _ in range(3)
            ],
            name="mini_detector_positional_conv",
        )

        self._cls_head = Dense(units=cls_num, activation="sigmoid")
        self._reg_head = Sequential(
            [
                Dense(units=256, activation="relu"),
                Dense(units=64, activation="relu"),
                Dense(
                    units=4, activation="sigmoid"
                ),  # use sigmoid instead of relu to restrict the coord between 0 and 1.
            ],
            name="mini_detector_regression_head",
        )
        self._pos_head = Sequential(
            [Dense(units=256, activation="relu"), Dense(units=2, activation="linear")]
        )

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
        return self._input_shape[0] * self._input_shape[1]

    @property
    def embedding_dim(self):
        return self._hidden_dim

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        cls_x = inputs
        cls_x = self._cls_conv(cls_x)
        cls_x = tf.reshape(
            cls_x, shape=(batch_size, self.sequence_length, self.embedding_dim)
        )
        cls_features = cls_x
        # shape of x after cls_head is (batch_size, sequence_length, cls_num)
        cls_scores = self._cls_head(cls_x)

        pos_query = gen_sineembed_for_position(
            tf.reshape(inputs, shape=(batch_size, 49, self.input_shape[-1])),
            d_model=self.input_shape[-1],
        )
        pos_query = tf.reshape(
            pos_query,
            shape=(
                batch_size,
                self.input_height,
                self.input_width,
                self.input_shape[-1],
            ),
        )
        pos_features = self._pos_conv(pos_query)
        pos_features = tf.reshape(
            pos_features, shape=(batch_size, self.sequence_length, self.embedding_dim)
        )

        pos_center_offset = self._pos_head(pos_features)
        pos_center_offset = tf.concat(
            [
                pos_center_offset,
                tf.zeros(shape=(batch_size, self.sequence_length, 2), dtype=tf.float32),
            ],
            axis=-1,
        )

        reg_x = inputs
        reg_x = self._reg_conv(reg_x)
        reg_x = tf.reshape(
            reg_x, shape=(batch_size, self.sequence_length, self.embedding_dim)
        )
        reg_features = reg_x
        # shape of x after reg_head is (batch_size, sequence_length, 4)
        bbox_coord = self._reg_head(reg_x) + pos_center_offset
        bbox_coord = sigmoid(bbox_coord)

        top_k = min(self._top_K, self.sequence_length)

        repr_cls_scores = tf.reduce_max(cls_scores, axis=-1)
        _, top_k_indices = tf.math.top_k(repr_cls_scores, k=top_k)
        cls_output = tf.gather(cls_features, top_k_indices, batch_dims=1)
        reg_output = tf.gather(reg_features, top_k_indices, batch_dims=1)
        top_k_centers = tf.gather(bbox_coord, top_k_indices, batch_dims=1)
        top_k_pos = gen_sineembed_for_position(
            pos_tensor=top_k_centers, d_model=2 * self._hidden_dim
        )

        top_k_proposals = tf.concat([cls_output, reg_output], axis=-1)  # 512
        all_proposals = tf.concat([cls_scores, bbox_coord], axis=-1)

        return top_k_proposals, top_k_centers, top_k_pos, all_proposals

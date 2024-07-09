import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Conv2D
from tensorflow.keras.models import Sequential

from .blocks.encoder_block import EncoderBlock
from .blocks.mini_detector import MiniDetector
from .blocks.decoder_block import DecoderBlock
from .utils.hungarian_algorithm import SingleTargetMatcher
from .utils.positional_encoding import gen_sineembed_for_position
from .utils.padding import padding_oh_labels
from .loss_functions.boxes_loss_functions import boxes_loss_v2
from .loss_functions.class_loss_functions import cls_loss


"""use IMDB-wiki https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ as training dataset
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

        self._resnet_backbone = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )

        self._reduce_dim = Conv2D(filters=hidden_dim, kernel_size=(1, 1))
        self._reduce_dim_2 = Conv2D(filters=hidden_dim // 2, kernel_size=(1, 1))
        self._mini_detector = MiniDetector(
            input_shape=(7, 7, hidden_dim),
            cls_num=num_cls,
            top_k=top_k,
            name="mini-detector",
        )
        self._num_encoders = num_encoder_blocks
        self._num_decoders = num_decoder_blocks

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
        self._decoder_blocks = [
            DecoderBlock(
                object_queries_shape=(top_k, hidden_dim),
                lambda_=0.5,
                name=f"DecoderBlock_{idx}",
            )
            for idx in range(num_decoder_blocks)
        ]

        self._hidden_dim = hidden_dim
        self._cls_ffn = tf.keras.layers.Dense(units=num_cls, activation="softmax")
        self._reg_ffn = tf.keras.layers.Dense(
            units=4, activation="sigmoid"
        )  # use sigmoid instead of relu to restrict the output between 0 and 1.
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

    def call(self, inputs):
        # extend for the batch dimension
        inputs = tf.cond(
            tf.equal(tf.rank(inputs), 3),
            lambda: tf.expand_dims(inputs, axis=0),
            lambda: inputs,
        )

        batch_size = tf.shape(inputs)[0]

        inputs = tf.ensure_shape(inputs, [None, *self._resnet_backbone.input.shape[1:]])
        inputs = tf.cast(inputs, dtype=tf.float32)

        tf.debugging.check_numerics(inputs, "NaN detected from inputs.")

        x = self._resnet_backbone(inputs)
        tf.debugging.check_numerics(x, "NaN detected from Backbone.")

        x = self._reduce_dim(x)  # reduce embedding dimension

        # positional encoding
        x = tf.reshape(x, shape=(batch_size, 49, self._hidden_dim))
        x += gen_sineembed_for_position(x, d_model=self._hidden_dim)

        x = self._encoder_blocks(x)
        encoder_output = x

        tf.debugging.check_numerics(x, "NaN detected from Encoder.")

        # reshape to (7, 7, -1) since there are conv layers in the mini-detector
        x = tf.reshape(x, shape=(batch_size, 7, 7, self._hidden_dim))
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

        encoder_output = tf.reshape(
            self._reduce_dim_2(tf.reshape(encoder_output, shape=(-1, 7, 7, 512))),
            shape=(-1, 49, 256),
        )  # embedding dimension from 512 to 256

        x = self._decoder_blocks[0](x, encoder_output, top_k_centers, top_k_pos)
        tf.debugging.check_numerics(x, "NaN detected from Decoder.")
        x = self._decoder_blocks[1](x, encoder_output, top_k_centers, top_k_pos)
        x = self._decoder_blocks[2](x, encoder_output, top_k_centers, top_k_pos)
        x = self._decoder_blocks[3](x, encoder_output, top_k_centers, top_k_pos)

        cls_x = x[..., : self._hidden_dim // 2]
        reg_x = x[..., self._hidden_dim // 2 :]
        cls_x = self.layer_norm1(cls_x)
        reg_x = self.layer_norm2(reg_x)

        cls_output = self._cls_ffn(cls_x)
        bbox_output = self._reg_ffn(reg_x)

        idx = tf.argmax(
            tf.reduce_max(cls_output, axis=-1), axis=-1, output_type=tf.int32
        )
        cls_output = tf.gather(cls_output, idx, batch_dims=1)
        bbox_output = tf.gather(bbox_output, idx, batch_dims=1)

        return cls_output, bbox_output, all_proposals


@tf.function
def train_one_step(model, optimizer, images, targets, num_cls: int = 8):
    single_tgt_matcher = SingleTargetMatcher(num_class=num_cls)

    tf.debugging.assert_rank(images, 4, name="check_rank_of_input_images")

    with tf.GradientTape(persistent=True) as tape:
        pred_cls, pred_boxes, mini_det_output = model(images)

        pred_cls_flat = tf.reshape(pred_cls, shape=(-1, num_cls))
        pred_boxes_flat = tf.reshape(pred_boxes, shape=(-1, 4))  # cxcy

        tgt_labels = tf.gather(targets, [0], axis=-1)
        tgt_oh_labels = tf.gather(targets, [1, 2, 3, 4, 5, 6, 7, 8], axis=-1)
        tgt_boxes = (
            tf.gather(targets, [9, 11, 10, 12], axis=-1) / 224.0
        )  # to min_x, min_y, max_x, max_y

        matched_idx = single_tgt_matcher(
            {
                "pred_obj_cls": mini_det_output[..., :num_cls],
                "pred_boxes": mini_det_output[..., num_cls:],
            },
            tgt_labels=tf.cast(tgt_labels, dtype=tf.int32),
            tgt_bbox=tgt_boxes,
        )

        matched_cls = tf.gather(
            mini_det_output[..., :num_cls], matched_idx, batch_dims=1
        )
        matched_cls = tf.reshape(matched_cls, shape=(-1, num_cls))
        matched_bbox = tf.gather(
            mini_det_output[..., num_cls:], matched_idx, batch_dims=1
        )
        matched_bbox = tf.reshape(matched_bbox, shape=(-1, 4))

        mini_det_loss = 0.5 * cls_loss(
            tgt_oh_labels, matched_cls
        ) + 0.5 * boxes_loss_v2(tgt_boxes, matched_bbox)
        model_loss = 0.5 * cls_loss(tgt_oh_labels, pred_cls_flat) + 0.5 * boxes_loss_v2(
            tgt_boxes, pred_boxes_flat
        )

    gradients_destr = tape.gradient(
        model_loss,
        model.trainable_variables,
    )
    gradients_destr = [
        grad if grad is not None else tf.zeros_like(var)
        for grad, var in zip(gradients_destr, model.trainable_variables)
    ]
    clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_destr]
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

    # mini_det_vars = model.get_layer(name='obj_det_split_transformer')._mini_detector.trainable_variables
    gradients_mini_det = tape.gradient(
        mini_det_loss,
        model.trainable_variables,
    )
    gradients_mini_det = [
        grad if grad is not None else tf.zeros_like(var)
        for grad, var in zip(gradients_mini_det, model.trainable_variables)
    ]
    clipped_gradients = [
        tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_mini_det
    ]
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

    del tape

    return mini_det_loss, model_loss


@tf.function
def validate(model, images, targets, num_cls: int = 8):
    single_tgt_matcher = SingleTargetMatcher(num_class=num_cls)

    tf.debugging.assert_rank(images, 4, name="check_rank_of_input_images")

    pred_cls, pred_boxes, mini_det_output = model(images)

    pred_cls_flat = tf.reshape(pred_cls, shape=(-1, num_cls))
    pred_boxes_flat = tf.reshape(pred_boxes, shape=(-1, 4))  # cxcy

    tgt_labels = tf.gather(targets, [0], axis=-1)
    tgt_oh_labels = tf.gather(targets, [1, 2, 3, 4, 5, 6, 7, 8], axis=-1)
    tgt_boxes = (
        tf.gather(targets, [9, 11, 10, 12], axis=-1) / 224.0
    )  # to min_x, min_y, max_x, max_y

    matched_idx = single_tgt_matcher(
        {
            "pred_obj_cls": mini_det_output[..., :num_cls],
            "pred_boxes": mini_det_output[..., num_cls:],
        },
        tgt_labels=tf.cast(tgt_labels, dtype=tf.int32),
        tgt_bbox=tgt_boxes,
    )

    matched_cls = tf.gather(mini_det_output[..., :num_cls], matched_idx, batch_dims=1)
    matched_cls = tf.reshape(matched_cls, shape=(-1, num_cls))
    matched_bbox = tf.gather(mini_det_output[..., num_cls:], matched_idx, batch_dims=1)
    matched_bbox = tf.reshape(matched_bbox, shape=(-1, 4))

    mini_det_loss = 0.5 * cls_loss(tgt_oh_labels, matched_cls) + 0.5 * boxes_loss_v2(
        tgt_boxes, matched_bbox
    )
    model_loss = 0.5 * cls_loss(tgt_oh_labels, pred_cls_flat) + 0.5 * boxes_loss_v2(
        tgt_boxes, pred_boxes_flat
    )

    return mini_det_loss, model_loss


@tf.function
def train_one_stepV2(
    model,
    optimizer,
    images,
    targets: tf.RaggedTensor,
    num_cls: int = 9,
    padding_to_k: int = 5,
    batch_size: int = 8,
):
    """Suppose that there are multiple objects in an image."""
    single_tgt_matcher = SingleTargetMatcher(num_class=num_cls)

    tf.debugging.assert_rank(images, 4, name="check_rank_of_input_images")

    with tf.GradientTape(persistent=True) as tape:
        pred_cls, pred_boxes, mini_det_output = model(images)

        pred_cls_flat = tf.reshape(pred_cls, shape=(-1, num_cls))
        pred_boxes_flat = tf.reshape(pred_boxes, shape=(-1, 4))  # cxcy

        tgt_labels = tf.gather(targets, [0], axis=-1).to_tensor(
            default_value=num_cls - 1, shape=(batch_size, padding_to_k, 1)
        )
        tgt_oh_labels = tf.gather(targets, [1, 2, 3, 4, 5, 6, 7, 8], axis=-1).to_tensor(
            default_value=0.0, shape=(batch_size, padding_to_k, num_cls - 1)
        )  # a class is for empty object
        tgt_oh_labels = padding_oh_labels(tgt_oh_labels)
        tgt_boxes = (
            tf.gather(targets, [9, 11, 10, 12], axis=-1).to_tensor(
                default_value=0.0, shape=(batch_size, padding_to_k, 4)
            )
            / 224.0
        )  # to min_x, min_y, max_x, max_y

        matched_idx = single_tgt_matcher(
            {
                "pred_obj_cls": mini_det_output[..., :num_cls],
                "pred_boxes": mini_det_output[..., num_cls:],
            },
            tgt_labels=tf.cast(tgt_labels, dtype=tf.int32),
            tgt_bbox=tgt_boxes,
        )

        matched_cls = tf.gather(
            mini_det_output[..., :num_cls], matched_idx, batch_dims=1
        )
        matched_cls = tf.reshape(matched_cls, shape=(-1, num_cls))
        matched_bbox = tf.gather(
            mini_det_output[..., num_cls:], matched_idx, batch_dims=1
        )
        matched_bbox = tf.reshape(matched_bbox, shape=(-1, 4))

        mini_det_loss = 0.5 * cls_loss(
            tgt_oh_labels, matched_cls
        ) + 0.5 * boxes_loss_v2(tgt_boxes, matched_bbox)
        model_loss = 0.5 * cls_loss(tgt_oh_labels, pred_cls_flat) + 0.5 * boxes_loss_v2(
            tgt_boxes, pred_boxes_flat
        )

    gradients_destr = tape.gradient(model_loss, model.trainable_variables)
    gradients_destr = [
        grad if grad is not None else tf.zeros_like(var)
        for grad, var in zip(gradients_destr, model.trainable_variables)
    ]
    clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_destr]
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

    # mini_det_vars = model.get_layer(name='obj_det_split_transformer')._mini_detector.trainable_variables
    gradients_mini_det = tape.gradient(mini_det_loss, model.trainable_variables)
    # tf.print(f"gradients: {gradients_mini_det}")
    gradients_mini_det = [
        grad if grad is not None else tf.zeros_like(var)
        for grad, var in zip(gradients_mini_det, model.trainable_variables)
    ]
    clipped_gradients = [
        tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_mini_det
    ]
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

    return mini_det_loss, model_loss


if __name__ == "__main__":
    model = ObjDetSplitTransformer(input_shape=(224, 224, 3), num_cls=8)

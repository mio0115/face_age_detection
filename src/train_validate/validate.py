import tensorflow as tf

from ..utils.hungarian_algorithm import SingleTargetMatcher
from ..model.loss_functions.boxes_loss_functions import boxes_loss_v2
from ..model.loss_functions.class_loss_functions import cls_loss


@tf.function
def validate(model, images, targets, num_cls: int = 8):
    single_tgt_matcher = SingleTargetMatcher(num_class=num_cls)

    tf.debugging.assert_rank(images, 4, name="check_rank_of_input_images")

    pred_cls, pred_boxes, mini_det_output = model(images)

    pred_cls_flat = tf.reshape(pred_cls, shape=(-1, num_cls))
    pred_boxes_flat = tf.reshape(pred_boxes, shape=(-1, 4))  # cxcy

    tgt_labels = tf.gather(targets, [0], axis=-1)
    tgt_oh_labels = tf.gather(targets, [1, 2, 3, 4, 5, 6, 7, 8], axis=-1)
    tgt_boxes = tf.gather(
        targets, [9, 11, 10, 12], axis=-1
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

    mini_det_loss = tf.constant(0.5, dtype=tf.float32) * cls_loss(
        tgt_oh_labels, matched_cls
    ) + tf.constant(0.5, dtype=tf.float32) * boxes_loss_v2(
        tgt_boxes, matched_bbox, alpha=0.0
    )
    model_loss = tf.constant(0.5, dtype=tf.float32) * cls_loss(
        tgt_oh_labels, pred_cls_flat
    ) + tf.constant(0.7, dtype=tf.float32) * boxes_loss_v2(
        tgt_boxes, pred_boxes_flat, alpha=0.0
    )
    box_loss = boxes_loss_v2(tgt_boxes, pred_boxes_flat, alpha=0.0)

    return mini_det_loss, model_loss, box_loss

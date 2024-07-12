import tensorflow as tf

from ...utils.bbox_utils import (
    from_cxcyhw_to_xyxy,
    smooth_l1_dist,
    get_iou,
    complete_iou,
)


@tf.function
def boxes_loss(tgt_boxes, pred_boxes, alpha=0.3, beta=0.3):
    invalid_hw = _penalty_of_weight_height(pred_boxes)
    pred_xyxy = from_cxcyhw_to_xyxy(pred_boxes)

    l1_loss = tf.reduce_mean(smooth_l1_dist(pred_xyxy, tgt_boxes), axis=-1)
    iou_loss = 1.0 - get_iou(pred_xyxy, tgt_boxes)

    return tf.reduce_mean(
        alpha * l1_loss + beta * iou_loss + (1 - alpha - beta) * invalid_hw
    )


@tf.function
def boxes_loss_v2(tgt_boxes, pred_boxes, alpha=0.5):
    """
    Compute the boxes loss between boundary box and target box.

    Args:
        pred_boxes: Coordinates of predicted boundary box. (cxcyhw)
        tgt_boxes : Coordinates of target boundary box. (xyxy)

    Return:
        tf.Tensor: loss of predicted boundary box.
    """
    pred_xyxy = from_cxcyhw_to_xyxy(pred_boxes)

    idx = tf.range(0, tf.shape(pred_boxes)[0])
    l1_loss = tf.gather(
        smooth_l1_dist(pred_xyxy, tgt_boxes), idx, batch_dims=1
    )  # take diagonal

    iou_loss = 1 - complete_iou(pred_xyxy, tgt_boxes)

    return tf.reduce_mean(alpha * l1_loss + (1 - alpha) * iou_loss)


@tf.function
def _penalty_of_weight_height(bbox, limit=1.0):
    """
    Give penalty to the illegal boundary box.

    Args:
        bbox: Coordinates of boundary box. (xyxy)

    Return:
        tf.Tensor: penalty of illegal boundary box.
    """
    # The function is to compute the penalty of illegal height and width
    # illegal height and width is the height and width that exceed 2 * center_y and 2 * center_x, respectively
    # bbox shape: (batch_size, seq_len, 4)
    center_x, center_y, height, width = tf.split(bbox, num_or_size_splits=4, axis=-1)

    penalty_height = tf.where(
        tf.less(center_y - height / 2, 0.0), tf.abs(center_y - height / 2), 0.0
    ) + tf.where(
        tf.greater(center_y + height / 2, limit), tf.abs(center_y + height / 2), 0.0
    )

    penalty_width = tf.where(
        tf.less(center_x - width / 2, 0.0), tf.abs(center_x - width / 2), 0.0
    ) + tf.where(
        tf.greater(center_x + width / 2, limit), tf.abs(center_x + width / 2), 0.0
    )

    return penalty_height + penalty_width

import math

import tensorflow as tf


@tf.function
def from_cxcyhw_to_xyxy(bbox_coord: tf.Tensor) -> tf.Tensor:
    """Transform the bbox coordinates
    from (center_x, center_y, height, width)
    to
    (min_x, min_y, max_x, max_y)
    we make min_x and min_y >= 0"""

    new_bbox_coord = tf.stack(
        [
            tf.maximum(bbox_coord[..., 0] - bbox_coord[..., 3] / 2, 0),
            tf.maximum(bbox_coord[..., 1] - bbox_coord[..., 2] / 2, 0),
            bbox_coord[..., 0] + bbox_coord[..., 3] / 2,
            bbox_coord[..., 1] + bbox_coord[..., 2] / 2,
        ],
        axis=-1,
    )

    return new_bbox_coord


@tf.function
def from_xyxy_to_cxcyhw(bbox_coord: tf.Tensor):
    """Transform the bbox coordinates
    from (min_x, min_y, max_x, max_y)
    to
    (center_x, center_y, height, width)"""

    new_bbox_coord = tf.stack(
        [
            (bbox_coord[..., 0] + bbox_coord[..., 2]) / 2,
            (bbox_coord[..., 1] + bbox_coord[..., 3]) / 2,
            bbox_coord[..., 3] - bbox_coord[..., 1],
            bbox_coord[..., 2] - bbox_coord[..., 0],
        ],
        axis=-1,
    )

    return new_bbox_coord


@tf.function
def smooth_l1_dist(bbox: tf.Tensor, tgt_bbox: tf.Tensor, delta: float = 1.0):
    # bbox     shape = (batch_size*seq_len, 4)
    # tgt_bbox shape = (batch_size, 4)
    seq_len, tgt_len = tf.shape(bbox)[0], tf.shape(tgt_bbox)[0]

    bbox = tf.broadcast_to(bbox[:, tf.newaxis, :], shape=(seq_len, tgt_len, 4))
    dist = tf.reduce_sum(bbox - tgt_bbox, axis=-1)

    loss = tf.where(
        tf.less(dist, delta), 0.5 * tf.square(dist) / delta, tf.abs(dist) - delta * 0.5
    )

    return loss


@tf.function
def get_iou(bbox: tf.Tensor, tgt_bbox: tf.Tensor):
    # The form of input coordinate is (left, upper, right, bottom)
    # Note that the zero point of an image is the left-upper corner

    seq_len, tgt_len = tf.shape(bbox)[0], tf.shape(tgt_bbox)[0]
    ext_bbox = tf.broadcast_to(bbox[:, tf.newaxis, :], shape=(seq_len, tgt_len, 4))

    # (x0, y0, x1, y1) = (min_x, min_y, max_x, max_y)
    # compute x0 and y0
    inter_mins = tf.maximum(
        tf.gather(ext_bbox, [0, 1], axis=-1), tf.gather(tgt_bbox, [0, 1], axis=-1)
    )
    # compute x1 and y1
    # permute the column to align it
    # in order to compute inter_wh
    inter_maxs = tf.minimum(
        tf.gather(ext_bbox, [2, 3], axis=-1), tf.gather(tgt_bbox, [2, 3], axis=-1)
    )

    inter_wh = tf.maximum(inter_maxs - inter_mins, 0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    # tf.debugging.assert_greater_equal(inter_area, tf.zeros_like(inter_area), "Inter Area should be all NON-NEGATIVE.")

    bbox_width = tf.maximum(bbox[..., 2] - bbox[..., 0], 0)
    bbox_height = tf.maximum(bbox[..., 3] - bbox[..., 1], 0)
    bbox_area = tf.broadcast_to(
        (bbox_width * bbox_height)[..., tf.newaxis], shape=(seq_len, tgt_len)
    )

    # tf.debugging.assert_greater(inter_area, tf.zeros_like(inter_area), "Pred Area should be all NON-NEGATIVE.")

    tgt_bbox_area = tf.multiply(
        tgt_bbox[..., 2] - tgt_bbox[..., 0], tgt_bbox[..., 3] - tgt_bbox[..., 1]
    )
    # tf.debugging.assert_greater(tgt_bbox_area, tf.zeros_like(inter_area), "Target Area should be all NON-NEGATIVE.")

    bbox_union_area = bbox_area + tgt_bbox_area - inter_area

    iou_val = inter_area / bbox_union_area

    return iou_val


@tf.function
def complete_iou(bbox: tf.Tensor, tgt_bbox: tf.Tensor) -> tf.Tensor:
    # Assume the form of bbox and tgt_bbox are both xyxy

    tgt_len = tf.shape(tgt_bbox)[0]

    bbox_cxcyhw = from_xyxy_to_cxcyhw(bbox)
    tgt_cxcyhw = from_xyxy_to_cxcyhw(tgt_bbox)

    iou = tf.gather(get_iou(bbox, tgt_bbox), tf.range(0, tgt_len), batch_dims=1)

    # print(f'bbox_cxcyhw: {bbox_cxcyhw}')
    # print(f'tgt_cxcyhw : {tgt_cxcyhw}')
    # print(f'iou: {iou}')

    # normalized distance between two boxes' center
    # normalized by the length of the diagonal of the minimum box that cover tgt_bbox and bbox
    # dist = tf.reduce_sum(tf.pow(tf.gather(tgt_cxcyhw - bbox_cxcyhw, [0, 1], axis=-1), 2), axis=-1) / tf.reduce_sum(tf.pow(big_box_max - big_box_min, 2), -1)
    center_diff = tgt_cxcyhw[..., :2] - bbox_cxcyhw[..., :2]
    # print(f'center_diff: {center_diff}')
    sq_center_dist = tf.reduce_sum(tf.square(center_diff), axis=-1)
    # print(f'sq_center_dist: {sq_center_dist}')

    big_box_cand = tf.stack([tgt_bbox, bbox], axis=-1)
    # print(f'big_box_cand: {big_box_cand}')
    big_box_min = tf.reduce_min(tf.gather(big_box_cand, [0, 1], axis=1), axis=-1)
    # print(f'big_box_min: {big_box_min}')
    big_box_max = tf.reduce_max(tf.gather(big_box_cand, [2, 3], axis=1), axis=-1)
    # print(f'big_box_max: {big_box_max}')
    big_box_diag_sq_len = tf.reduce_sum(tf.square(big_box_max - big_box_min), axis=-1)
    # print(f'big_box_diag_sq_len: {big_box_diag_sq_len}')

    norm_dist = sq_center_dist / big_box_diag_sq_len
    # print(f'norm_dist: {norm_dist}')

    # Aspect Ratio computes as width / height
    tgt_aspect_ratio = tf.tanh(tgt_cxcyhw[..., 3] / tgt_cxcyhw[..., 2])
    bbox_aspect_ratio = tf.tanh(
        bbox_cxcyhw[..., 3] / tf.maximum(bbox_cxcyhw[..., 2], 1e-6)
    )
    sq_aspect_ratio_diff = tf.square(tgt_aspect_ratio - bbox_aspect_ratio)

    aspect_ratio = 4.0 / tf.square(math.pi) * sq_aspect_ratio_diff
    alpha = tf.where(tf.less(iou, 0.5), 0.0, aspect_ratio / (1 - iou + aspect_ratio))

    return iou - norm_dist - alpha * aspect_ratio

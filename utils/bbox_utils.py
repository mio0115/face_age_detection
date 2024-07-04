import math

import tensorflow as tf


@tf.function
def from_cxcyhw_to_xyxy(bbox_coord: tf.Tensor) -> tf.Tensor:
    """ Transform the bbox coordinates 
        from (center_x, center_y, height, width) 
        to 
        (min_x, min_y, max_x, max_y)
        we make min_x and min_y >= 0"""
    
    new_bbox_coord = tf.stack(
        [tf.maximum(bbox_coord[..., 0] - bbox_coord[..., 3]/2, 0),
         tf.maximum(bbox_coord[..., 1] - bbox_coord[..., 2]/2, 0),
         bbox_coord[..., 0] + bbox_coord[..., 3]/2,
         bbox_coord[..., 1] + bbox_coord[..., 2]/2,],
        axis=-1
    )
    
    return new_bbox_coord

@tf.function
def smooth_l1_dist(bbox: tf.Tensor, tgt_bbox: tf.Tensor, delta: float=1.0):
    # bbox     shape = (batch_size*seq_len, 4)
    # tgt_bbox shape = (batch_size, 4)
    seq_len, tgt_len = tf.shape(bbox)[0], tf.shape(tgt_bbox)[0]

    bbox = tf.broadcast_to(bbox[:, tf.newaxis, :], shape=(seq_len, tgt_len, 4))
    dist = tf.reduce_sum(bbox - tgt_bbox, axis=-1)

    loss = tf.where(tf.less(dist, delta), 0.5*tf.square(dist), tf.abs(dist)-0.5)
    
    return loss

@tf.function
def get_iou(bbox: tf.Tensor, tgt_bbox: tf.Tensor):
    seq_len, tgt_len = tf.shape(bbox)[0], tf.shape(tgt_bbox)[0]
    ext_bbox = tf.broadcast_to(bbox[:, tf.newaxis, :], shape=(seq_len, tgt_len, 4))

    # (x0, y0, x1, y1) = (min_x, min_y, max_x, max_y)
    # compute x0 and y0
    inter_mins = tf.maximum(
        tf.gather(ext_bbox, [0, 1], axis=-1),
        tf.gather(tgt_bbox, [0, 1], axis=-1)
    )
    # compute x1 and y1
    # permute the column to align it
    # in order to compute inter_wh
    inter_maxs = tf.minimum(
        tf.gather(ext_bbox, [2, 3], axis=-1),
        tf.gather(tgt_bbox, [2, 3], axis=-1)
    )
    
    inter_wh = tf.maximum(inter_maxs - inter_mins, 0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    #tf.debugging.assert_greater_equal(inter_area, tf.zeros_like(inter_area), "Inter Area should be all NON-NEGATIVE.")

    bbox_width = tf.maximum(bbox[..., 2] - bbox[..., 0], 0)
    bbox_height = tf.maximum(bbox[..., 3] - bbox[..., 1], 0)
    bbox_area = tf.broadcast_to((bbox_width * bbox_height)[..., tf.newaxis], shape=(seq_len, tgt_len))

    #tf.debugging.assert_greater(inter_area, tf.zeros_like(inter_area), "Pred Area should be all NON-NEGATIVE.")

    tgt_bbox_area = tf.multiply(tgt_bbox[..., 2] - tgt_bbox[..., 0], tgt_bbox[..., 3] - tgt_bbox[..., 1])
    #tf.debugging.assert_greater(tgt_bbox_area, tf.zeros_like(inter_area), "Target Area should be all NON-NEGATIVE.")

    bbox_union_area = bbox_area + tgt_bbox_area - inter_area

    iou_val = inter_area / bbox_union_area

    return iou_val

@tf.function
def complete_iou(bbox: tf.Tensor, tgt_bbox: tf.Tensor) -> tf.Tensor:
    bbox_cxcyhw = bbox
    bbox_xyxy = from_cxcyhw_to_xyxy(bbox_cxcyhw)

    iou = get_iou(bbox_xyxy, tgt_bbox)
    
    bbox_cx, bbox_cy = bbox_cxcyhw[..., 0], bbox_cxcyhw[..., 1]

    tgt_cx, tgt_cy = (tgt_bbox[..., 0] + tgt_bbox[..., 2]) / 2, (tgt_bbox[..., 1] + tgt_bbox[..., 3]) / 2
    tgt_h, tgt_w = tgt_bbox[..., 3] - tgt_bbox[..., 1], tgt_bbox[..., 2] - tgt_bbox[..., 0]

    # distance between two center of boxes over length of the diagonal of the minimum box that cover tgt_bbox and bbox
    dist = (tf.pow(tgt_cx - bbox_cx, 2) + \
            tf.pow(tgt_cy - bbox_cy, 2)) / \
        (tf.pow(tgt_bbox[..., 2] - bbox_xyxy[..., 0], 2) + \
         tf.pow(tgt_bbox[..., 3] - bbox_xyxy[..., 1], 2))
    
    aspect_ratio = 4 / tf.pow(math.pi, 2) * tf.pow(tf.tanh(tgt_w / tgt_h) - tf.tanh(bbox_cxcyhw[..., 3] / bbox_cxcyhw[..., 2]), 2)
    alpha = tf.where(tf.less(iou, 0.5), 0., aspect_ratio / (1 - iou + aspect_ratio))

    return iou - dist - alpha * aspect_ratio
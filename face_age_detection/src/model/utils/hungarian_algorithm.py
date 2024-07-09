import tensorflow as tf
import numpy as np

# from scipy.optimize import linear_sum_assignmen

from ..utils.bbox_utils import smooth_l1_dist, from_cxcyhw_to_xyxy, get_iou

"""We pair up an object in ground truth with an object proposal from mini-detector.

Input : Tensor (shape = (batch_size, sequence_length, number_of_classes))
Output: Tensor (shape = (batch_size, sequence_length, 2), the last 2 is the (matching ground truth index, loss))

Step 1) Subtract row minimum
Step 2) Subtract column minimum
Step 3) Create additional zeros
    Create additional zeros through minus the minimum value among elements not covered by the lines
    Then add the value to the elements covered by two lines

After each step, we apply the check to the matrix
Check ) Cover zeros with a minimum number of lines (could be row or column)
        In step Check, we observe that whether the number of lines equal to the rank
"""


def _to_cost_matrix(inputs: tf.Tensor):
    return tf.reduce_max(inputs, axis=[1, 2])[:, tf.newaxis, tf.newaxis] - inputs


def _step1(cost_matrix: tf.Tensor):
    return cost_matrix - tf.reduce_min(cost_matrix, axis=2)[:, :, tf.newaxis]


def _step2(cost_matrix: tf.Tensor):
    return cost_matrix - tf.reduce_min(cost_matrix, axis=1)[:, tf.newaxis, :]


def _step3(cost_matrix: tf.Tensor, lines: list):
    cross_idx = []
    n = cost_matrix.shape[0]
    for r in lines["R"]:
        for c in lines["C"]:
            cross_idx.append((r, c))

    uncovered_idx = []
    for r in range(n):
        for c in range(n):
            if r not in lines["R"] and c not in lines["C"]:
                uncovered_idx.append((r, c))
    min_val = tf.reduce_min(tf.gather_nd(cost_matrix, indices=uncovered_idx))

    cost_matrix = tf.tensor_scatter_nd_add(cost_matrix, cross_idx, min_val)
    cost_matrix = tf.tensor_scatter_nd_sub(cost_matrix, uncovered_idx, min_val)
    return cost_matrix


def _cover_zeros(cost_matrix: tf.Tensor) -> list:
    """Greedy
    To derive the minimal line set that cover the zeros in the matrix
    """
    matrix = cost_matrix.numpy()
    n = cost_matrix.shape[0]
    lines = {"R": [], "C": []}

    while np.count_nonzero(matrix) < n**2:
        nonzero_cnt = np.concatenate(
            (np.count_nonzero(matrix, axis=1), np.count_nonzero(matrix, axis=0))
        )
        idx = np.argmin(nonzero_cnt)
        lines["R" if idx < n else "C"].append(idx % n)

        if idx >= n:  # column
            idx -= n
            matrix[:, idx] += 1
        else:
            matrix[idx, :] += 1
    return lines


def _padding_to_square(matrix: tf.Tensor) -> tf.Tensor:
    _, seq_len, cls_num = matrix.shape

    if seq_len < cls_num:
        padding = [[0, 0], [0, cls_num - seq_len], [0, 0]]
    else:
        padding = [[0, 0], [0, 0], [0, seq_len - cls_num]]

    return tf.pad(matrix, padding)


def _find_matching(cost_matrix):
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    idx = list(zip(row_idx, col_idx))

    return idx


def bipartite_matching(inputs):
    assert tf.rank(inputs) == 3
    batch_size, seq_len, cls_num = inputs.shape

    if seq_len != cls_num:
        inputs = _padding_to_square(inputs)
        cls_num = inputs.shape[-1]

    cost_matrix = _to_cost_matrix(inputs)
    results = []

    for b in range(batch_size):
        cost_matrix = cost_matrix[b].numpy()
        lines = []

        cost_matrix = _step1(cost_matrix)
        lines = _cover_zeros(cost_matrix)
        if len(lines["R"] + lines["C"]) == cls_num:
            results.append(_find_matching(cost_matrix))
            continue

        cost_matrix = _step2(cost_matrix)
        lines = _cover_zeros(cost_matrix)
        if len(lines["R"] + lines["C"]) == cls_num:
            results.append(_find_matching(cost_matrix))
            continue

        lines = _cover_zeros(cost_matrix)
        while len(lines["R"] + lines["C"]) < cls_num:
            cost_matrix = _step3(cost_matrix, lines)
            lines = _cover_zeros(cost_matrix)
        results.append(_find_matching(cost_matrix))

    return


def maximum_prob_assignment(inputs) -> tf.Tensor:
    batch_size, seq_len, cls_num = np.shape(inputs)
    object_allocation = np.zeros(shape=(-1, seq_len, 2), dtype=np.float32)

    if cls_num < seq_len:
        inputs = _padding_to_square(inputs)

    cost_matrix = _to_cost_matrix(inputs)
    for b in range(batch_size):
        row_idx, col_idx = linear_sum_assignment(cost_matrix[b])
        indices = np.array(sorted([(row, col) for row, col in zip(row_idx, col_idx)]))

        object_allocation[b, :, 0] = indices[:, 1]
        object_allocation[b, :, 1] = tf.gather_nd(inputs[b], indices=indices)

    return object_allocation


@tf.function
def tf_linear_sum_assignment(cost_matrix):
    return tf.numpy_function(
        func=linear_sum_assignment, inp=[cost_matrix], Tout=[tf.int64, t]
    )


class HungarianMatcher(tf.keras.layers.Layer):
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        num_cls: int = 8,
    ):
        super(HungarianMatcher, self).__init__()

        self._cost_cls = cost_class
        self._cost_bbox = cost_bbox
        self._cost_giou = cost_giou
        self._num_cls = num_cls

    def call(self, outputs, targets):
        batch_size, num_objects = tf.shape(outputs["pred_obj_cls"])[:2]

        out_prob = tf.nn.sigmoid(
            tf.reshape(
                outputs["pred_obj_cls"], shape=[batch_size * num_objects, self._num_cls]
            ),
            axis=-1,
        )
        out_bbox = tf.reshape(
            outputs["pred_boxes"], shape=[batch_size * num_objects, 4]
        )

        tgt_ids = tf.concat([tgt["labels"] for tgt in targets], axis=-1)
        tgt_bbox = tf.concat([tgt["boxes"] for tgt in targets], axis=-1)

        param = {"neg_coeff": 0.25, "power": 2.0}
        neg_cost_class = (
            (1 - param["neg_coeff"])
            * (out_prob ** param["power"])
            * tf.nn.log(-(1 - out_prob + 1e-8))
        )
        pos_cost_class = (
            param["neg_coeff"]
            * ((1 - out_prob) ** param["power"])
            * tf.nn.log(-(out_prob + 1e-8))
        )
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = tf.norm()

        return tf.stop_gradient()


class SingleTargetMatcher(tf.keras.layers.Layer):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 1.0,
        cost_iou: float = 1.0,
        num_class: int = 8,
    ):
        super(SingleTargetMatcher, self).__init__()

        self._cost_cls = cost_class
        self._cost_bbox = cost_bbox
        self._cost_iou = cost_iou
        self._num_cls = num_class

    def call(
        self, outputs, tgt_labels, tgt_bbox, is_xyxy=tf.constant(False, dtype=tf.bool)
    ):
        batch_size, num_objects = (
            tf.shape(outputs["pred_obj_cls"])[0],
            tf.shape(outputs["pred_obj_cls"])[1],
        )

        out_prob = tf.sigmoid(
            tf.reshape(
                outputs["pred_obj_cls"], shape=(batch_size * num_objects, self._num_cls)
            )
        )
        out_bbox = tf.reshape(
            outputs["pred_boxes"], shape=(batch_size * num_objects, 4)
        )

        alpha, gamma = 0.25, 2.0
        neg_cost_cls = (
            (1 - alpha) * (out_prob**gamma) * (-tf.math.log(1 - out_prob + 1e-8))
        )
        pos_cost_cls = (
            alpha * ((1 - out_prob) ** gamma) * (-tf.math.log(out_prob + 1e-8))
        )

        cost_cls = tf.gather(pos_cost_cls, tgt_labels, axis=1) - tf.gather(
            neg_cost_cls, tgt_labels, axis=1
        )
        cost_cls = tf.squeeze(cost_cls, axis=-1)

        # check the form of target_bbox coordinate
        out_bbox = tf.cond(
            is_xyxy,
            true_fn=lambda: out_bbox,
            false_fn=lambda: from_cxcyhw_to_xyxy(out_bbox),
        )
        l1_bbox_cost = smooth_l1_dist(bbox=out_bbox, tgt_bbox=tgt_bbox)
        # Note that iou is not "cost"
        # Higher IoU represents more fit to target.
        iou_bbox_cost = -1 * get_iou(bbox=out_bbox, tgt_bbox=tgt_bbox)

        # shape: (batch_size*num_objects, target_num,)
        # Cost between objects and targets across all batches
        cost = (
            self._cost_bbox * l1_bbox_cost
            + self._cost_cls * cost_cls
            + self._cost_iou * iou_bbox_cost
        )

        cost = tf.reshape(cost, shape=(batch_size, num_objects, -1))

        tgt_sizes = [1] * 8 # 8 is the batch size

        return [
            tf.argmin(cost[batch_idx], axis=0, output_type=tf.int32)
            for batch_idx, cost in enumerate(
                tf.split(cost, num_or_size_splits=tgt_sizes, axis=-1)
            )
        ]

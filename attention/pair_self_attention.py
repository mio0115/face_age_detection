import tensorflow as tf
from tensorflow.keras.layers import Dense


class PairSelfAttention(tf.keras.layers.Layer):
    def __init__(self, input_shape: tuple[int], heads_num: int):
        super(PairSelfAttention, self).__init__()
        self._input_shape = input_shape
        self._heads_num = heads_num

        self._proj_to_query = Dense(units=self.per_head_dim, use_bias=False)
        self._proj_to_key = Dense(units=self.per_head_dim, use_bias=False)
        self._proj_to_value = Dense(units=self.per_head_dim, use_bias=False)
    
    @property
    def input_shape(self):
        return self._input_shape
    
    @property
    def heads_num(self):
        return self._heads_num
    
    @property
    def sequence_length(self):
        return self._input_shape[0]
    
    @property
    def input_embedding_dim(self):
        return self._input_shape[-1]
    
    @property
    def per_head_dim(self):
        return self._input_shape[-1] // self._heads_num

    def _split_heads(self, x):
        batch_size = tf.shape(x)[0]

        x = tf.reshape(x, shape=(batch_size, self.sequence_length, self.heads_num, self.per_head_dim))
        x = tf.transpose(x, perm=[0, 2, 1, 3])

        return x

    def call(self, content_embedding, obj_pos_encoding, top_k_centers):
        """ Implmentation based on DESTR: Object Detection with Split Transformer 
        Instead of self-attention, authors use pair self-attention.
        The steps of pair self-attention are as following:
        1. Pairs up those components of input feature map.
        2. Compute a2 score.
        3. Compute o2 score.
        """
        batch_size = tf.shape(content_embedding)[0]
        
        # shape = (batch_size, sequence_length, input_embedding_dim)
        x_to_query_key = self._split_heads(content_embedding + obj_pos_encoding)
        x_to_value = self._split_heads(content_embedding)

        """ The following block is to find indices of pairs based on their IoU. 
        Component a is paired up with Component a' if their IoU is larger than other components
        To compute A2, we need to pair up indices of (a, b) and (a', b')
        """
        pairs = _get_pairs(top_k_centers) # shape (batch_size, head_nums, seq_len, 4)
        idx_pairs_l = tf.stack(
            [tf.broadcast_to(pairs[:, :, tf.newaxis, 0], shape=[batch_size, self.sequence_length, self.sequence_length]),  # index for query
             tf.broadcast_to(pairs[:, tf.newaxis, :, 0], shape=[batch_size, self.sequence_length, self.sequence_length])], # index for key
             axis=-1
        )
        idx_pairs_r = tf.stack(
            [tf.broadcast_to(pairs[:, :, tf.newaxis, 1], shape=[batch_size, self.sequence_length, self.sequence_length]),
             tf.broadcast_to(pairs[:, tf.newaxis, :, 1], shape=[batch_size, self.sequence_length, self.sequence_length])],
             axis=-1
        )
        
        query = self._proj_to_query(x_to_query_key)
        key = self._proj_to_key(x_to_query_key)
        value = self._proj_to_value(x_to_value)
        
        """ We compute A2(a, b) = <q_{pi a}, k_{pi b}> + <q_{pi a'}, k_{pi b'}> """
        a2 = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2]))

        a2_l = tf.gather_nd(
            a2, 
            tf.broadcast_to(idx_pairs_l[:, tf.newaxis, ...], 
                            shape=[batch_size, self.heads_num, self.sequence_length, self.sequence_length, 2]), 
            batch_dims=2
        )
        a2_r = tf.gather_nd(
            a2, 
            tf.broadcast_to(idx_pairs_r[:, tf.newaxis, ...], 
                            shape=[batch_size, self.heads_num, self.sequence_length, self.sequence_length, 2]),
            batch_dims=2
        )

        a2 = tf.keras.activations.softmax((a2_l + a2_r) / tf.sqrt(2 * tf.cast(self.input_embedding_dim, dtype=tf.float32)))
        # shape = (batch_size, heads_num, sequence_length, per_head_dim)
        o2 = tf.matmul(a2, value)
        o2 = tf.transpose(o2, perm=[0, 2, 1, 3])

        return tf.reshape(o2, shape=[batch_size, self.sequence_length, -1])


@tf.function
def _get_pairs(top_k_coord: tf.Tensor):
    """According to DESTR, pair self-attention has better performance than self-attention.
    For each object query, we only take the pair which has the highest IoU.
    Then order the pair by their L1-distance decreasingly.
    """
    # top_k_coord is shape (batch_size, num_objects, 4)
    top_k_coord = tf.ensure_shape(top_k_coord, shape=[None, None, 4])

    batch_size = tf.shape(top_k_coord)[0]
    num_objects = tf.shape(top_k_coord)[1]
    
    # top-left x, bottom-right y, bottom-right x, top-left y
    bbox_coord = tf.stack(
        [top_k_coord[..., 0] - top_k_coord[..., 3] / 2,
         top_k_coord[..., 1] - top_k_coord[..., 2] / 2,
         top_k_coord[..., 0] + top_k_coord[..., 3] / 2,
         top_k_coord[..., 1] + top_k_coord[..., 2] / 2,
         ], axis=-1
    )

    bbox_coord1 = tf.expand_dims(bbox_coord, axis=2)
    bbox_coord2 = tf.expand_dims(bbox_coord, axis=1)

    # IoU = intersection_area / union_area
    inter_mins = tf.maximum(bbox_coord1[..., :2], bbox_coord2[..., :2])
    inter_maxs = tf.minimum(bbox_coord1[..., 2:], bbox_coord2[..., 2:])
    inter_wh = tf.maximum(inter_maxs - inter_mins, 0)

    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    bbox_area = tf.multiply(bbox_coord[..., 2]-bbox_coord[..., 0], bbox_coord[..., 3]-bbox_coord[..., 1])
    bbox_area1 = tf.expand_dims(bbox_area, axis=-1)
    bbox_area2 = tf.expand_dims(bbox_area, axis=-2)

    bbox_union_area = bbox_area1 + bbox_area2 - inter_area
    # the IoU between two same objects is 1, we do not want that kind of pair.
    bbox_iou = inter_area / bbox_union_area - tf.eye(num_objects, batch_shape=[batch_size])

    # turn the indices from [[0, 2, 1]] to [[[0, 0], [1, 2], [2, 1]]]
    pair_idx = tf.stack([
        tf.broadcast_to(tf.range(start=0, limit=num_objects)[tf.newaxis, ...], shape=(batch_size, num_objects)), 
        tf.argmax(bbox_iou, axis=-1, output_type=tf.int32)], 
        axis=-1
    )

    # compute the L1-distance for each bbox
    bbox_l1 = (top_k_coord[..., 2]-top_k_coord[..., 0]) + (top_k_coord[..., 3]-top_k_coord[..., 1])
    # get the corresponding L1-distance of each bbox.
    # for example, the index pair for 0 is [0, 1], then we have [L1-distance for bbox0, L1-distance for bbox1]
    bbox_l1_pair = tf.stack([
            bbox_l1,
            tf.gather(bbox_l1, pair_idx[..., 1], batch_dims=1)
        ], axis=-1
    )

    # determine the order of each pair based on their L1-score
    # the index of the bbox with larger L1-distance is the first one index in the pair
    correct_order = tf.where((bbox_l1_pair[..., 0] >= bbox_l1_pair[..., 1])[..., tf.newaxis], pair_idx, tf.reverse(pair_idx, axis=[-1]))

    return correct_order
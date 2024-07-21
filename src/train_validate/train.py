import logging, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import json
import gc
import argparse
from distutils.util import strtobool

from ..model.destr_model import build_model
from ..model.loss_functions.class_loss_functions import cls_loss
from ..model.loss_functions.boxes_loss_functions import boxes_loss_v2
from ..utils.data_loader import load_data_tfrecord
from ..utils.hungarian_algorithm import SingleTargetMatcher
from ..utils.padding import padding_oh_labels
from .validate import validate


# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# tf.keras.mixed_precision.set_global_policy(policy)


def train_model(
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    num_train_samples: int,
    num_valid_samples: int,
    shuffle_buffer: int,
    to_checkpoint_dir: str,
    to_dataset: str,
    to_loss_records: str,
    num_class: int,
    input_shape: tuple[int],
    load_from_ckpt: str,
    num_encoder_blocks: int = 6,
    num_decoder_blocks: int = 6,
    top_k: int = 5,
):
    model = build_model(
        input_shape=input_shape,
        num_cls=num_class,
        num_encoder_blocks=num_encoder_blocks,
        num_decoder_blocks=num_decoder_blocks,
        top_k=top_k,
    )

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, directory=to_checkpoint_dir, max_to_keep=3
    )

    if load_from_ckpt:
        status = checkpoint.restore(checkpoint_manager.latest_checkpoint)

    loss_history = {"train_loss": (0, 0), "valid_loss": (0, 0)}

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizers = {
        "mini_det": tf.keras.optimizers.Adam(
            learning_rate=learning_rate * 0.1, clipvalue=3.0
        ),
        "model": tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=3.0),
    }
    # optimizers["mini_det"] = tf.keras.mixed_precision.LossScaleOptimizer(
    #    optimizers["mini_det"]
    # )
    # optimizers["model"] = tf.keras.mixed_precision.LossScaleOptimizer(
    #    optimizers["model"]
    # )
    full_dataset = load_data_tfrecord(path_to_tfrecord=to_dataset)

    train_progress_bar = tf.keras.utils.Progbar(num_train_samples)
    valid_progress_bar = tf.keras.utils.Progbar(num_valid_samples)

    dataset = full_dataset.shuffle(buffer_size=shuffle_buffer)
    train_dataset = dataset.take(count=num_train_samples * batch_size)
    valid_dataset = dataset.skip(count=num_train_samples * batch_size).take(
        count=num_valid_samples * batch_size
    )

    for epoch_idx in range(num_epochs):
        train_batches = train_dataset.shuffle(buffer_size=shuffle_buffer).batch(
            batch_size=batch_size, drop_remainder=True
        )
        valid_batches = valid_dataset.shuffle(buffer_size=shuffle_buffer).batch(
            batch_size=batch_size, drop_remainder=True
        )

        total_md_loss, total_loss, step, total_box_loss = 0, 0, 0, 0
        for batch in train_batches:
            logits, coord, label, oh_label = batch

            mini_det_loss, model_loss, box_loss = train_one_step(
                model,
                optimizers,
                tf.reshape(
                    tf.cast(tf.io.decode_raw(logits, tf.uint8), tf.float32),
                    shape=(-1,) + input_shape,
                ),
                tf.concat([label[..., tf.newaxis], oh_label, coord / 224], axis=-1),
            )
            total_md_loss += mini_det_loss.numpy()
            total_loss += model_loss.numpy()
            total_box_loss += box_loss.numpy()

            step += 1
            train_progress_bar.update(step)
            if step == num_train_samples:
                avg_mini_det_loss = total_md_loss / num_train_samples
                avg_model_loss = total_loss / num_train_samples
                avg_box_loss = total_box_loss / num_train_samples
                break
        loss_history["train_loss"] = (avg_mini_det_loss, avg_model_loss, avg_box_loss)

        gc.collect()

        total_md_loss, total_loss, step, total_box_loss = 0, 0, 0, 0
        for batch in valid_batches:
            logits, coord, label, oh_label = batch

            mini_det_loss, model_loss, box_loss = validate(
                model,
                tf.reshape(
                    tf.cast(tf.io.decode_raw(logits, tf.uint8), tf.float32),
                    shape=(-1,) + input_shape,
                ),
                tf.concat([label[..., tf.newaxis], oh_label, coord / 224], axis=-1),
            )
            total_md_loss += mini_det_loss.numpy()
            total_loss += model_loss.numpy()
            total_box_loss += box_loss.numpy()

            step += 1
            valid_progress_bar.update(step)
            if step == num_valid_samples:
                avg_mini_det_loss = total_md_loss / num_valid_samples
                avg_model_loss = total_loss / num_valid_samples
                avg_box_loss = total_box_loss / num_valid_samples
                break
        loss_history["valid_loss"] = (avg_mini_det_loss, avg_model_loss, avg_box_loss)

        # Save parameters after each epoch
        checkpoint_manager.save()

        tf.keras.backend.clear_session()
        gc.collect()

        print(
            f"""epoch {epoch_idx+1:>2}: \n
            \t train_loss: {loss_history["train_loss"][0]:.4f} {loss_history["train_loss"][1]:.4f}, {loss_history["train_loss"][2]:.4f}
            \t valid loss: {loss_history["valid_loss"][0]:.4f} {loss_history["valid_loss"][1]:.4f}, {loss_history["valid_loss"][2]:.4f}"""
        )
        with open(to_loss_records, mode="w") as fout:
            print(
                f"""epoch {epoch_idx+1:>2}: \n
            \t train_loss: {loss_history["train_loss"][0]:.4f} {loss_history["train_loss"][1]:.4f},
            \t valid loss: {loss_history["valid_loss"][0]:.4f} {loss_history["valid_loss"][1]:.4f}""",
                file=fout,
            )

    return model


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

        matched_cls = tf.gather(
            mini_det_output[..., :num_cls], matched_idx, batch_dims=1
        )
        matched_cls = tf.reshape(matched_cls, shape=(-1, num_cls))
        matched_bbox = tf.gather(
            mini_det_output[..., num_cls:], matched_idx, batch_dims=1
        )
        matched_bbox = tf.reshape(matched_bbox, shape=(-1, 4))

        mini_det_loss = tf.constant(0.5, dtype=tf.float32) * cls_loss(
            tgt_oh_labels, matched_cls
        ) + tf.constant(0.5, dtype=tf.float32) * boxes_loss_v2(
            tgt_boxes, matched_bbox, alpha=0.0
        )
        model_loss = tf.constant(0.3, dtype=tf.float32) * cls_loss(
            tgt_oh_labels, pred_cls_flat
        ) + tf.constant(0.7, dtype=tf.float32) * boxes_loss_v2(
            tgt_boxes, pred_boxes_flat, alpha=0.0
        )
        box_loss = boxes_loss_v2(tgt_boxes, pred_boxes_flat, alpha=0.0)

    gradients_destr = tape.gradient(
        model_loss,
        model.trainable_variables,
    )
    gradients_destr = [
        grad if grad is not None else tf.zeros_like(var)
        for grad, var in zip(gradients_destr, model.trainable_variables)
    ]
    optimizer["model"].apply_gradients(zip(gradients_destr, model.trainable_variables))

    # mini_det_vars = model.get_layer(name='obj_det_split_transformer')._mini_detector.trainable_variables
    gradients_mini_det = tape.gradient(
        mini_det_loss,
        model.trainable_variables,
    )
    gradients_mini_det = [
        grad if grad is not None else tf.zeros_like(var)
        for grad, var in zip(gradients_mini_det, model.trainable_variables)
    ]
    optimizer["mini_det"].apply_gradients(
        zip(gradients_mini_det, model.trainable_variables)
    )

    del tape

    return mini_det_loss, model_loss, box_loss


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
        tgt_boxes = tf.gather(targets, [9, 11, 10, 12], axis=-1).to_tensor(
            default_value=0.0, shape=(batch_size, padding_to_k, 4)
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
    # clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_destr]
    optimizer.apply_gradients(zip(gradients_destr, model.trainable_variables))

    # mini_det_vars = model.get_layer(name='obj_det_split_transformer')._mini_detector.trainable_variables
    gradients_mini_det = tape.gradient(mini_det_loss, model.trainable_variables)
    gradients_mini_det = [
        grad if grad is not None else tf.zeros_like(var)
        for grad, var in zip(gradients_mini_det, model.trainable_variables)
    ]
    # clipped_gradients = [
    #    tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_mini_det
    # ]
    optimizer.apply_gradients(zip(gradients_mini_det, model.trainable_variables))

    return mini_det_loss, model_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        help="learning rate of optimizer",
        default=0.000005,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        dest="batch_size",
        help="batch size for dataset",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        help="number of training epochs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--num_train_samples",
        dest="num_train_samples",
        help="number of training samples in epoch",
        default=20000,
        type=int,
    )
    parser.add_argument(
        "--num_valid_samples",
        dest="num_valid_samples",
        help="number of validate samples in epoch",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--shuffle_buffer",
        dest="shuffle_buffer",
        help="dataset shuffle buffer size",
        default=20000,
        type=int,
    )
    parser.add_argument(
        "--to_checkpoint",
        dest="path_to_checkpoints",
        help="path to checkpoint directory",
        default="/workspace/models/checkpoints",
    )
    parser.add_argument(
        "--to_dataset",
        dest="path_to_dataset",
        help="path to dataset",
        default="/workspace/data/tfrecords",
    )
    parser.add_argument(
        "--to_loss_records",
        dest="path_to_loss_records",
        help="path to store loss records",
        default="/workspace/models/loss.txt",
    )
    parser.add_argument(
        "--restore_from_ckpt",
        dest="restore_from_ckpt",
        help="restore from latest checkpoints",
        default=False,
        type=lambda x: strtobool(x),
    )
    parser.add_argument(
        "--num_cls", dest="num_cls", help="number of class to classify", default=8
    )
    parser.add_argument(
        "--input_shape",
        dest="input_shape",
        help="shape of input",
        default=(224, 224, 3),
    )
    parser.add_argument(
        "--num_encoder_blocks",
        dest="num_encoder_blocks",
        help="number of encoder blocks",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num_decoder_blocks",
        dest="num_decoder_blocks",
        help="number of decoder blocks",
        default=6,
        type=int,
    )
    parser.add_argument(
        "--top_k",
        dest="top_k",
        help="k objects took in mini-detector",
        default=5,
        type=int,
    )

    args = parser.parse_args()
    train_model(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_train_samples=args.num_train_samples,
        num_valid_samples=args.num_valid_samples,
        shuffle_buffer=args.shuffle_buffer,
        to_checkpoint_dir=args.path_to_checkpoints,
        to_dataset=args.path_to_dataset,
        to_loss_records=args.path_to_loss_records,
        num_class=args.num_cls,
        input_shape=args.input_shape,
        load_from_ckpt=args.restore_from_ckpt,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        top_k=args.top_k,
    )

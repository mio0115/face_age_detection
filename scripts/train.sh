#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
python -m src.train_validate.train --restore_from_ckpt=True --num_train_samples=20000 --to_checkpoint="/workspace/models/checkpoints_norm_720" --num_decoder_blocks=6 --num_encoder_blocks=6
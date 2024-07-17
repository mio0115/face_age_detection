#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
python -m src.train_validate.train --restore_from_ckpt=False --num_train_samples=15000 --to_checkpoint="/workspace/models/checkpoints_norm_717" --num_decoder_blocks=6 --num_encoder_blocks=3
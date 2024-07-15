#!/bin/bash

python -m src.train_validate.train --restore_from_ckpt=False --num_train_samples=15000 --to_checkpoint="/workspace/models/checkpoints_norm" --num_decoder_blocks=6 --num_encoder_blocks=3
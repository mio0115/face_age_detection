#!/bin/bash

python -m src.train_validate.train --num_train_samples=10 --num_valid_samples=5 --shuffle_buffer=10 --restore_from_ckpt=True
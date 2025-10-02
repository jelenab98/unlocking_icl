#!/bin/bash

# high burstiness
python train.py --use_wandb --run_name DTD_3311_bursty --dataset dtd --data_mode sequence --bursty_image_format 3 3 1 1 --resize_dim 128

# high burstiness + instCopy
python train.py --use_wandb --run_name DTD_3311_burstyInstCopy --dataset dtd --data_mode sequence_with_copy --bursty_image_format 3 3 1 1 --resize_dim 128

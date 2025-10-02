#!/bin/bash

# high burstiness
python train.py --use_wandb --run_name Omniglot_L3H1_3311_bursty --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 --attention_metrics --calculate_label_to_image_value --n_layer 3 --n_head 1

# high burstiness + instCopy
python train.py --use_wandb --run_name Omniglot_L3H1_3311_burstyInstCopy --dataset omniglot --data_mode sequence_with_copy --bursty_image_format 3 3 1 1 --attention_metrics --calculate_label_to_image_value --n_layer 3 --n_head 1

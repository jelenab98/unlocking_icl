#!/bin/bash

# high burstiness
python train.py --use_wandb --run_name Omniglot_3311_bursty --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 

# high burstiness + instCopy
python train.py --use_wandb --run_name Omniglot_3311_burstyInstCopy --dataset omniglot --data_mode sequence_with_copy --bursty_image_format 3 3 1 1 

# low burstiness
python train.py --use_wandb --run_name Omniglot_11111111_bursty --dataset omniglot --data_mode sequence --bursty_image_format 1 1 1 1 1 1 1 1 

# low burstiness + instCopy
python train.py --use_wandb --run_name Omniglot_11111111_burstyInstCopy --dataset omniglot --data_mode sequence_with_copy --bursty_image_format 1 1 1 1 1 1 1 1 

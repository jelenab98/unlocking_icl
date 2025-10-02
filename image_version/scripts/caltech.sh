#!/bin/bash

# high burstiness
python train.py --use_wandb --run_name Caltech_3311_bursty --dataset caltech --data_mode sequence --bursty_image_format 3 3 1 1 

# high burstiness + instCopy
python train.py --use_wandb --run_name Caltech_3311_burstyInstCopy --dataset caltech --data_mode sequence_with_copy --bursty_image_format 3 3 1 1 

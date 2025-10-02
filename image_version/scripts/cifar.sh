#!/bin/bash

# high burstiness
python train.py --use_wandb --run_name Cifar_3311_bursty --dataset cifar --data_mode sequence --bursty_image_format 3 3 1 1 

# high burstiness + instCopy
python train.py --use_wandb --run_name Cifar_3311_burstyInstCopy --dataset cifar --data_mode sequence_with_copy --bursty_image_format 3 3 1 1 

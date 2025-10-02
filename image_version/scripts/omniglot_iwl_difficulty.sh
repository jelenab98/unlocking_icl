#!/bin/bash

# high burstiness on 200 classes
python train.py --use_wandb --run_name Omniglot_3311_bursty_numClasses200 --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 --reduced_factor 200

# high burstiness on 400 classes
python train.py --use_wandb --run_name Omniglot_3311_bursty_numClasses400 --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 --reduced_factor 400

# high burstiness on 600 classes
python train.py --use_wandb --run_name Omniglot_3311_bursty_numClasses600 --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 --reduced_factor 600

# high burstiness on 800 classes
python train.py --use_wandb --run_name Omniglot_3311_bursty_numClasses800 --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 --reduced_factor 800

# high burstiness on 1000 classes
python train.py --use_wandb --run_name Omniglot_3311_bursty_numClasses1000 --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 --reduced_factor 1000


# high burstiness on 1600 classes
python train.py --use_wandb --run_name Omniglot_3311_bursty_numClasses1600 --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1



# high burstiness on 200 classes + label swapping
python train.py --use_wandb --run_name Omniglot_3311_bursty_numClasses200_labelSwap --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 --reduced_factor 200 --p_noisy_labels 0.2

# high burstiness on 400 classes + label swapping
python train.py --use_wandb --run_name Omniglot_3311_bursty_numClasses400_labelSwap --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 --reduced_factor 400 --p_noisy_labels 0.2

# high burstiness on 600 classes + label swapping
python train.py --use_wandb --run_name Omniglot_3311_bursty_numClasses600_labelSwap --dataset omniglot --data_mode sequence --bursty_image_format 3 3 1 1 --reduced_factor 600 --p_noisy_labels 0.2

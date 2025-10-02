# Unlocking In-Context Learning for Natural Datasets Beyond Language Modelling (**GCPR 2025 Best Paper Honorable Mention**)

---

Official PyTorch implementation of our GCPR 2025 paper:  [Unlocking In-Context Learning for Natural Datasets Beyond Language Modelling](https://arxiv.org/abs/2501.06256)

---

## Repository Overview
This repository is organized into two main parts:

- **`image_version`**: Experiments on vision datasets (Omniglot, CIFAR, Caltech, DTD).  
- **`EEG_version`**: Experiments on EEG BCI classification.  

---

## Image Dataset Experiments

### 1. Prepare datasets
Create symbolic links to the datasets under `image_version/data`.  
- Datasets should be structured with one folder per class, named after the class.  

### 2. Logging
The code uses **Weights & Biases (W&B)** for experiment tracking.  
- Disabling W&B will result in *no metrics being logged during training*.  

### 3. Running experiments
Example experiment scripts can be found in `image_version/scripts`.  

To run training with **burstiness** and **instance copies** in the input sequences, use:  
```bash
python train.py --use_wandb --run_name Omniglot_3311_burstyInstCopy \
                --dataset omniglot --data_mode sequence_with_copy \
                --bursty_image_format 3 3 1 1
```

## EEG BCI Experiments

### 1. Prepare datasets
Create symbolic links to the datasets under `EEG_version/data`.  
- Datasets should be structured with one folder per class, named after the class.  

### 2. Logging
As with image experiments, **W&B** is required for logging results. 

### 3. Running experiments
EEG scripts and details will be released soon!

## Citation
```
@InProceedings{UnlockingICL_2025_GCPR,
    author    = {Bratulić, Jelena and Mittal, Sudhanshu and Hoffmann, David T. and B{\"o}hm, Samuel and Schirrmeister, Robin T. and Ball, Tonio and Rupprecht, Christian and Brox, Thomas},
    title     = {Unlocking In-Context Learning for Natural Datasets Beyond Language Modelling},
    booktitle = {Proceedings of the DAGM German Conference on Pattern Recognition},
    year      = {2025},
}
```
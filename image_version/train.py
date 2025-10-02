import torch
import wandb
import random
import argparse
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from typing import List

from constants import DATASET_MEAN_STD
from utils import save_model, get_args
from icl_dataset import ICLDataset
from trainer import Trainer
from GPT2 import GPT2


def get_lr_lambda(warmup_steps=4000, total_steps=20000):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return (warmup_steps ** 0.5) * (current_step ** -0.5)
    return lr_lambda


def get_datasets(args):
    dataset_mean, dataset_std = DATASET_MEAN_STD[args.dataset_name]
    sequences_2w4s = None
    sequences_4w2s = None
    
    if args.dataset_name == "omniglot":
        train_transforms = transforms.Compose([transforms.Resize(args.resize_dim), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.Resize(args.resize_dim), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])

        if args.load_presampled_sequences:
            sequences_2w4s = "presampled_icl_sequences/omniglot_2w4s.txt"
            sequences_4w2s = "presampled_icl_sequences/omniglot_4w2s.txt"
    
    elif args.dataset_name == "cifar":
        
        train_transforms = transforms.Compose([
            transforms.Resize((args.resize_dim, args.resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
             ])

        test_transforms = transforms.Compose([
            transforms.Resize((args.resize_dim, args.resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
             ])
    
    elif args.dataset_name == "caltech":
        
        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop((args.resize_dim, args.resize_dim), scale=(0.5, 1.5)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std)
                ])

        test_transforms = transforms.Compose([
            transforms.Resize((args.resize_dim, args.resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
             ])
    
    elif args.dataset_name == "dtd":

        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop((args.resize_dim, args.resize_dim), scale=(0.5, 1.5)),
                transforms.ToTensor(),
                transforms.Normalize(dataset_mean, dataset_std)
                ])

        test_transforms = transforms.Compose([
            transforms.Resize((args.resize_dim, args.resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
             ])
        
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported.")


    dataset_train = ICLDataset(dataset_name=args.dataset_name, split="train", reduced_factor=args.reduced_factor, data_mode=args.data_mode,
                               p_bursty=args.p_bursty, bursty_image_format=args.bursty_image_format, p_noisy_labels=args.p_noisy_labels,
                               fixed_number_of_samples=args.fixed_number_of_samples, transforms=train_transforms)

    dataset_IWL = ICLDataset(dataset_name=args.dataset_name, split="test_iwl", reduced_factor=args.reduced_factor, data_mode="sequence",
                            bursty_image_format=args.bursty_image_format, fixed_number_of_samples=None, transforms=test_transforms)
    
    dataset_ICL_2w4s = ICLDataset(dataset_name=args.dataset_name, split="test_icl", data_mode="sequence", icl_image_format=[4, 4],
                            fixed_number_of_samples=10000, transforms=test_transforms, sequences_file=sequences_2w4s)
    
    dataset_ICL_4w2s = ICLDataset(dataset_name=args.dataset_name, split="test_icl", data_mode="sequence", icl_image_format=[2, 2, 2, 2],
                            fixed_number_of_samples=10000, transforms=test_transforms, sequences_file=sequences_4w2s)

    return dataset_train, dataset_IWL, dataset_ICL_2w4s, dataset_ICL_4w2s


def get_dataloaders(args):
    dataset_train, dataset_IWL, dataset_ICL_2w4s, dataset_ICL_4w2s = get_datasets(args)

    num_classes = dataset_train.num_classes
    seq_len = dataset_train.seq_len

    train_dataloader = DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    test_iwl_dataloader = DataLoader(dataset_IWL, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers, pin_memory=True)

    test_icl_dataloaders = {
        "2w4s": DataLoader(dataset_ICL_2w4s, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers, pin_memory=True),
        "4w2s": DataLoader(dataset_ICL_4w2s, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_num_workers, pin_memory=True)}

    return train_dataloader, test_iwl_dataloader, test_icl_dataloaders, num_classes, seq_len


def main(args):

    if args.use_wandb:
        wandb.init(project=args.project_name, name=args.run_name, config=vars(args), dir="./wandb")
        wandb.config.update(vars(args))

    # Setting the seeds 
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # Creating the datasets and dataloaders
    train_dataloader, test_iwl_dataloader, test_icl_dataloaders, num_classes, seq_len = get_dataloaders(args)
    
    # Creating the model, optimizer and scheduler
    model = GPT2(
                block_size=seq_len,
                vocab_size=num_classes,
                emb_dim=args.emb_dim,
                n_head=args.n_head,
                n_layer=args.n_layer,
                attn_drop=args.dropout,
                resid_drop=args.dropout,
                mlp_drop=args.dropout,
                resnet_embed=args.resnet_embed,
                resize_dim=args.resize_dim,
                initialization=args.initialization,
                std=args.std,
                a=args.init_a,
                b=args.init_b,
            ).cuda()

    print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(),betas=(args.beta1, args.beta2), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda(args.warmup_steps, args.total_steps))

    # Initialize the trainer
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path)

        model.load_state_dict(checkpoint["params"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
       
        trainer = Trainer(args, model, train_dataloader, test_iwl_dataloader, test_icl_dataloaders, optimizer, scheduler, torch.nn.CrossEntropyLoss(), iter_num=checkpoint["iter_num"])
    else:
        trainer = Trainer(args, model, train_dataloader, test_iwl_dataloader, test_icl_dataloaders, optimizer, scheduler, torch.nn.CrossEntropyLoss())
    
    try:
        trainer.run()
        if args.use_wandb and args.experiment_mode == "train":
            save_model(model, optimizer, scheduler, trainer.iter_num, f"{args.run_name}_final", description="Final model")
    except KeyboardInterrupt:
        print(f"Interrupted training at iteration {trainer.iter_num}.")
        if args.use_wandb and args.experiment_mode == "train":
            save_model(model, optimizer, scheduler, trainer.iter_num, f"{args.run_name}_model_tmp", description=f"Model at iter {trainer.iter_num}")

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    args = get_args()
    main(args)

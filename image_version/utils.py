import os
import torch
import wandb
import argparse
import matplotlib.pyplot as plt

from constants import DATASET_MEAN_STD


def save_model(model, optimizer, scheduler, iter_num, run_name, description):
    save_dict = {}
    save_dict["params"] = model.state_dict()
    save_dict["optimizer"] = optimizer.state_dict()
    save_dict["scheduler"] = scheduler.state_dict()
    save_dict["iter_num"] = iter_num
    
    model_file = f"./{run_name}_model_it{iter_num}.pth"
    
    torch.save(save_dict, model_file)
    
    artifact = wandb.Artifact(run_name, type='model', description=description)
    artifact.add_file(model_file)
    wandb.log_artifact(artifact)
    
    os.remove(model_file)

    return model_file


def visualize_sequence(batch, args, save_name="batches"):
    images, targets = batch
    figure = plt.figure(figsize=(16, 10))
    dataset_mean, dataset_std = DATASET_MEAN_STD[args.dataset_name]
    
    if dataset_mean is not None and dataset_std is not None:
        dataset_mean = torch.tensor(dataset_mean).view(-1, 1, 1)
        dataset_std = torch.tensor(dataset_std).view(-1, 1, 1)
        images_to_visualize = [(images[idx][0]*dataset_std + dataset_mean).clamp(0, 1).squeeze().numpy().transpose(1, 2, 0) for idx in range(0, len(images), 2)]
    else:
        images_to_visualize = [images[idx][0].squeeze().numpy().transpose(1, 2, 0) for idx in range(0, len(images), 2)]

    labels_to_visualize = targets[0, ::2]
   
    for idx in range(len(images_to_visualize)):
        plt.subplot(3, 3, idx +1)
        plt.imshow(images_to_visualize[idx])
        plt.xlabel(labels_to_visualize[idx].item())

    wandb.log({save_name: plt})
    plt.close()


def get_args():
    args = argparse.ArgumentParser()

    # experiment mode
    args.add_argument('--experiment_mode', choices=('train', 'eval'), default='train', help='Mode for the experiment.')

    # wandb args
    args.add_argument('--use_wandb', action='store_true', help="Flag for using WandB. If you don't use WandB, no metrics or logs will be saved!")
    args.add_argument('--project_name', type=str, default="unlocking_icl", help="Name of the WandB project.")
    args.add_argument('--run_name', type=str, default="base_train", help='Name of the run.')
    
    # evaluation
    args.add_argument('--resume_path', type=str, default=None, help='Path to the checkpoint which will be restored to resume training or to evaluate.')
    args.add_argument('--resume_iter', type=int, default=None, help='Iteration number of the checkpoint which will be restored to resume training or to evaluate.')
    
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--log_interval', type=int, default=1500, help="Interval for the train metric printing. In steps.")
    args.add_argument('--eval_interval', type=int, default=1500, help="Interval for the ICL and IWL evaluation frequency. In steps.")
    args.add_argument('--total_steps', type=int, default=500000, help="Total number of steps.")
   
    # gpt2 config
    args.add_argument('--n_layer', type=int, default=12, help="Number of layers.")
    args.add_argument('--n_head', type=int, default=8, help="Number of heads.")
    args.add_argument('--dropout', type=float, default=0.1, help="Dropout value.")
    args.add_argument('--emb_dim', type=int, default=64, help="Embedding dimension.")
    args.add_argument('--resnet_embed', type=str, default="SmallResNet2Blocks", choices=("SmallResNet2Blocks", "ResNet18Pretrained"), help="Image embedder.")

    args.add_argument('--initialization', type=str, default="truncate", choices=("default", "truncate"), help="Initialization of the model.")
    args.add_argument('--std', type=float, default=0.02, help="Standard deviation for the initialization.")
    args.add_argument('--init_a', type=float, default=-0.04, help="Min value for truncation initialization.")
    args.add_argument('--init_b', type=float, default=0.04, help="Max value for truncation initialization.")
    
    # optimizator 
    args.add_argument('--lr', type=float, default=6e-4, help="Learning rate.")
    args.add_argument('--beta1', type=float, default=0.9, help="Beta for Adam.")
    args.add_argument('--beta2', type=float, default=0.99, help="Beta for Adam.")
    args.add_argument('--eps', type=float, default=1e-8, help="Epsilon for Adam.")
    args.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay. ICL wasn't emerging when it was used.")
    args.add_argument('--warmup_steps', type=int, default=15000, help="Warmup steps which are needed for the lr reach lr value set with flags.")
    args.add_argument('--clip_grad', type=float, default=1.0)
    
    # dtd
    args.add_argument('--resize_dim', type=int, default=64, help="Input image dimension.")
    
    # dataset
    args.add_argument('--dataset_name', choices=('cifar', 'dtd', 'caltech', 'omniglot'), default='omniglot', help="Type of the dataset")
    args.add_argument('--class_split_mode', type=str, choices=('base_novel', 'all', 'random', ), default='base_novel', help="split of classes. Base-novel uses predefined splits, all uses all classes, random uses random split into base novel with num_random_classes used for novel class")
    args.add_argument('--p_bursty', type=float, default=0.9, help="Percentage of bursty sequence in total data. Burstiness 1 - only ICL and no IWL, Burstiness 0 - only IWL and no ICL.")
    args.add_argument('--bursty_image_format', nargs='+', type=int, default=[3, 3, 1, 1], help="Format of the sequence.")
    args.add_argument('--fixed_number_of_samples', type=int, default=64000, help="Fixed number of samples for the train loader. Loader is in theory infitive due to the large number of possible sequences, but for better control over the epoch number, we can set it to some number (64000). If None, will use the number of images.")
    args.add_argument('--reduced_factor', type=int, default=0, help="Factor for reducing number of classes when using predefined base-novel split.")
    args.add_argument('--data_mode', type=str, default="sequence", help="Type of IWL task.",
                      choices=("sequence", "sequence_with_copy")) 
    args.add_argument('--p_noisy_labels', type=float, default=0.0, help="Percentage of noisy labels in the dataset.")
    args.add_argument('--load_presampled_sequences', action='store_true', help="If set, will load presampled sequences instead of computing them on the fly.")

    # dataloader
    args.add_argument('--train_batch_size', type=int, default=16, help="Train batch size.")
    args.add_argument('--test_batch_size', type=int, default=100, help="Test batch size.")
    args.add_argument('--num_workers', type=int, default=8, help="Number of workers.")
    args.add_argument('--test_num_workers', type=int, default=8, help="Number of workers for test.")

    # attention maps analysis
    args.add_argument('--attention_metrics', action='store_true', help="Calculate the attention progress metrics.")
    args.add_argument("--num_maps_to_visualize", type=int, default=1, help="Number of attention maps to visualize.")
    args.add_argument('--calculate_label_to_image_value', action='store_true', help="Calculate label to image attention value in training for layer 1. Only used for the analysis of the smaller model.")

    return args.parse_args()

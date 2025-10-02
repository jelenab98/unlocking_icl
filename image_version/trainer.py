import wandb
import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from tqdm import tqdm

from utils import save_model, visualize_sequence

class Trainer:
    def __init__(self, 
                 args, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 test_iwl_loader: DataLoader, 
                 test_icl_loaders, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler.LambdaLR, 
                 criterion: nn.CrossEntropyLoss, 
                 iter_num: int = 0):
        
        self.model = model
        self.train_loader = train_loader
        self.test_iwl_loader = test_iwl_loader
        self.test_icl_loaders = test_icl_loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.iter_num = iter_num

        self.model = self.model.to(self.device)

    def run(self):
        if self.args.experiment_mode == "train":
            print("Starting the train loop with the following config: ")
            print(vars(self.args))
            self.train()
        else:
            print("Starting the eval loop with the following config: ")
            print(vars(self.args))
            self.model.eval()
            self.test_icl(self.iter_num)
            self.test_iwl(self.iter_num)

    def calculate_label_to_image_value(self, attns):
        diag_indices = torch.arange(1, self.model.block_size, step=2).cuda()
        diag_mask = torch.zeros(self.model.block_size, self.model.block_size).cuda()
        diag_mask[diag_indices, diag_indices - 1] = 1
        attns_layer_1 = attns[1][:, 0, :, :]
        label_to_image_values = (attns_layer_1 * diag_mask).sum(dim=(-1, -2)) / diag_mask.sum()  # (B, T)
        return label_to_image_values.mean().item()

    def train(self):
        self.model.train()
        epoch_counter = 0
        self.optimizer.zero_grad()
        
        while self.iter_num < self.args.total_steps:
            train_loss = 0
            correct = 0
            for batch_idx, batch in enumerate(tqdm(self.train_loader)):  

                if batch_idx == 0 and epoch_counter < 5:
                    visualize_sequence(batch, self.args, save_name=f"batches")

                logits, target, loss, attns = self.model(batch)
                record_to_plot = {}
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                train_loss += loss.item()
                self.iter_num += 1

                correct = self.accuracy(logits, target)
                if self.args.calculate_label_to_image_value:
                    label_to_image_values = self.calculate_label_to_image_value(attns)
                else:
                    label_to_image_values = 0
                
                if self.args.use_wandb:
                    wandb.log({"train_loss": train_loss / (batch_idx + 1),
                                "lr": self.scheduler.get_last_lr()[0],
                                "train_acc": 100. * correct / len(target),
                                "label_to_image_per_step": label_to_image_values,})

                if batch_idx % self.args.log_interval == 0:
                    print("Train set {}: Train loss: {:.4f}, Lr: {:.6f}, Accuracy: {}/{} ({:.0f}%)".format(self.iter_num, train_loss/(batch_idx+1), 
                                                                                                                self.scheduler.get_last_lr()[0], correct, len((target)),
                                                                                                                100.*correct/len(target)))
                
                if self.iter_num  % self.args.eval_interval == 0:
                    print(f"Performing validation with current iter num {self.iter_num} for ", self.args.run_name)
                    self.test_icl(self.iter_num)
                    self.test_iwl(self.iter_num)
                    self.model.train()

                if self.iter_num > self.args.total_steps:
                    break

            epoch_counter += 1
            print(f"Finished epoch {epoch_counter} with current num iters: {self.iter_num}")

    def test_iwl(self, epoch):
        if self.test_iwl_loader is None:
            print("No IWL loader provided, skipping IWL test.")
            return
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_iwl_loader)):

                if batch_idx < 3 and self.iter_num < 5000:
                    visualize_sequence(batch, self.args, save_name=f"iwl_batches")

                logits, target, loss, _ = self.model(batch)
                test_loss += loss.item()
                correct += self.accuracy(logits, target)
                total += len(target)

        print(f"Test IWL {epoch}: Average loss: {test_loss / (batch_idx + 1):.4f}, Accuracy: {correct}/{total} ({100. * correct / total:.2f}%)")
        
        if self.args.use_wandb:
            wandb.log({"iwl_loss": test_loss / (batch_idx + 1),
                       "iwl_acc":100. * correct / total,
                       "iwl_step": epoch})   

    def test_icl(self, epoch):
        self.model.eval()
        metrics = {}
        for test_mode, data_loader in self.test_icl_loaders.items():
            metrics[f"icl_loss_{test_mode}"] = 0
            metrics[f"icl_01_{test_mode}"] = 0
            total = 0
            tmp_metrics = {}
            for n_layer in range(self.args.n_layer):
                tmp_metrics[f"attn_maps_layer_{n_layer}_imgs"] = 0
                tmp_metrics[f"attn_maps_layer_{n_layer}_labels"] = 0
                tmp_metrics[f"attn_maps_layer_{n_layer}_imgs_all"] = 0
                tmp_metrics[f"attn_maps_layer_{n_layer}_labels_all"] = 0

                tmp_metrics[f"attn_maps_layer_{n_layer}_averaged_labels_to_image"] = 0
                tmp_metrics[f"attn_maps_layer_{n_layer}_averaged_image_to_image"] = 0
                tmp_metrics[f"attn_maps_layer_{n_layer}_averaged_image_to_image_all"] = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(data_loader)):

                    if batch_idx < 3 and self.iter_num < 5000:
                        visualize_sequence(batch, self.args, save_name=f"{test_mode}_batches")
                    
                    all_targets = batch[1].cuda()
                    logits, target, loss, atts = self.model(batch)

                    # get different logits bins for prediction
                    valid_indices = (all_targets.unique()[1:]).cpu().numpy().tolist()   # remove -1
                    valid_mappings = {}
                    for idx, value in enumerate(valid_indices):
                        valid_mappings[idx] = value

                    if self.args.attention_metrics:
                        # calculate average attn scores for all samples in a batch
                        query_labels = all_targets[:, -1].unsqueeze(1)
                        query_indices_batch, query_indices_imgs = (all_targets[:, :-1] == query_labels).nonzero(as_tuple=True)
                        query_indices_labels = query_indices_imgs + 1

                        query_indices_all_batch, query_indices_all_imgs = (all_targets[:, :-1] != -1).nonzero(as_tuple=True)
                        query_indices_all_labels = query_indices_all_imgs + 1

                        diag_indices = torch.arange(1, self.model.block_size, step=2).cuda()
                        diag_mask = torch.zeros(self.model.block_size, self.model.block_size).cuda()
                        diag_mask[diag_indices, diag_indices - 1] = 1

                        image_indices = torch.arange(0, self.model.block_size - 1, step=2, device=all_targets.device)
                        img_mask = torch.tril(torch.ones((self.model.block_size // 2, self.model.block_size // 2), device=all_targets.device), diagonal=-1)
                        same_class_mask = (all_targets[:, :-1:2] == all_targets[:, -1].unsqueeze(-1)).float()

                        for idx_layer, attn in enumerate(atts):
                            final_img_scores = attn[:, :, -1, :]
                            image_to_image_vals = (final_img_scores[query_indices_batch, :, query_indices_imgs]).sum(dim=0)
                            image_to_label_vals = (final_img_scores[query_indices_batch, :, query_indices_labels]).sum(dim=0)
                            image_to_image_vals_all = (final_img_scores[query_indices_all_batch, :, query_indices_all_imgs]).sum(dim=0)
                            image_to_label_vals_all = (final_img_scores[query_indices_all_batch, :, query_indices_all_labels]).sum(dim=0)
                            
                            tmp_metrics[f"attn_maps_layer_{idx_layer}_imgs"] += image_to_image_vals
                            tmp_metrics[f"attn_maps_layer_{idx_layer}_labels"] += image_to_label_vals
                            tmp_metrics[f"attn_maps_layer_{idx_layer}_imgs_all"] += image_to_image_vals_all
                            tmp_metrics[f"attn_maps_layer_{idx_layer}_labels_all"] += image_to_label_vals_all

                            tmp_metrics[f"attn_maps_layer_{idx_layer}_averaged_labels_to_image"] += ((attn * diag_mask).sum(dim=(-1, -2)) / diag_mask.sum()).sum(dim=0)
                            
                            image_attention_scores = attn.index_select(2, image_indices).index_select(3, image_indices)

                            masked_image_attention_scores = image_attention_scores * img_mask.unsqueeze(0).unsqueeze(0)
                            same_class_attention_scores = masked_image_attention_scores * same_class_mask.unsqueeze(1).unsqueeze(1)
                            
                            tmp_metrics[f"attn_maps_layer_{idx_layer}_averaged_image_to_image_all"] += (masked_image_attention_scores.sum(dim=(-1, -2)) / img_mask.shape[-1]).sum(dim=0)
                            tmp_metrics[f"attn_maps_layer_{idx_layer}_averaged_image_to_image"] += (same_class_attention_scores.sum(dim=(-1, -2)) / same_class_mask.shape[-1]).sum(dim=0)
                    
                    logits_01 = logits[:, valid_indices]

                    target = target
                    metrics[f"icl_loss_{test_mode}"] += loss.item()
                    metrics[f"icl_01_{test_mode}"] += self.accuracy(logits_01, target, valid_mappings)
                    total += len(target)
                 
            
            metrics[f"icl_loss_{test_mode}"] /= (batch_idx + 1)
            metrics[f"icl_01_{test_mode}"] *= 100. / total

            metrics[f"icl_{test_mode}_step"] = epoch

            if self.args.attention_metrics:
                for idx_layer in range(self.args.n_layer):
                    layer_metrics_imgs = tmp_metrics[f"attn_maps_layer_{idx_layer}_imgs"] / total
                    layer_metrics_labels = tmp_metrics[f"attn_maps_layer_{idx_layer}_labels"] / total
                    layer_metrics_imgs_all = tmp_metrics[f"attn_maps_layer_{idx_layer}_imgs_all"] / total
                    layer_metrics_labels_all = tmp_metrics[f"attn_maps_layer_{idx_layer}_labels_all"] / total
          
                    layer_metrics_label_to_image = tmp_metrics[f"attn_maps_layer_{idx_layer}_averaged_labels_to_image"] / total
                    layer_metrics_image_to_image = tmp_metrics[f"attn_maps_layer_{idx_layer}_averaged_image_to_image"] / total
                    layer_metrics_image_to_image_all = tmp_metrics[f"attn_maps_layer_{idx_layer}_averaged_image_to_image_all"] / total

                    for idx_head in range(self.args.n_head):
                        metrics[f"attn_maps_{test_mode}_layer_{idx_layer}_head_{idx_head}_imgs"] = layer_metrics_imgs[idx_head].item()
                        metrics[f"attn_maps_{test_mode}_layer_{idx_layer}_head_{idx_head}_labels"] = layer_metrics_labels[idx_head].item()
                        metrics[f"attn_maps_{test_mode}_layer_{idx_layer}_head_{idx_head}_imgs_all"] = layer_metrics_imgs_all[idx_head].item()
                        metrics[f"attn_maps_{test_mode}_layer_{idx_layer}_head_{idx_head}_labels_all"] = layer_metrics_labels_all[idx_head].item()
        
                        metrics[f"attn_maps_{test_mode}_layer_{idx_layer}_head_{idx_head}_averaged_labels_to_image"] = layer_metrics_label_to_image[idx_head].item()
                        metrics[f"attn_maps_{test_mode}_layer_{idx_layer}_head_{idx_head}_averaged_image_to_image"] = layer_metrics_image_to_image[idx_head].item()
                        metrics[f"attn_maps_{test_mode}_layer_{idx_layer}_head_{idx_head}_averaged_image_to_image_all"] = layer_metrics_image_to_image_all[idx_head].item()
            
            print(f'Test ICL-{test_mode} {epoch}: Average loss: {metrics[f"icl_loss_{test_mode}"]:.4f}, Accuracy: {metrics[f"icl_01_{test_mode}"]:.2f}%')
        
        if self.args.use_wandb:
            wandb.log(metrics)
        
    def accuracy(self, output, target, label_mappings=None):
        pred = output.max(1, keepdim=True)[1]
        if label_mappings is not None:
            for key, value in label_mappings.items():
                pred[pred == key] = value
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct

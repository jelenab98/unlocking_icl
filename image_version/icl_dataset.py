import torch
import random
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from constants import *


class ICLDataset(Dataset):
    def __init__(self,
                 dataset_name: str = "omniglot", 
                 split: str = "train", 
                 class_split_mode: str = "base_novel",
                 reduced_factor: int = 0,
                 seed: int = 42,
                 data_mode: str = "sequence",
                 n_support: int = 8,
                 n_query: int = 1,
                 p_bursty: float = 0.0,
                 icl_image_format: list = [4, 4],
                 bursty_image_format: list = [3, 3, 1, 1],
                 p_noisy_labels: float = 0.0,
                 fixed_number_of_samples: int = None,
                 transforms=None,
                 sequences_file=None) -> None:
        super().__init__()

        self.root = Path(PATHS_TO_DATASETS[dataset_name])
        self.split = split                          # train, test_iwl, test_icl
        self.class_split_mode = class_split_mode    # base_novel, all, random (based on a seed)
        self.data_mode = data_mode                  # sequence, sequence_with_copy
        
        self.p_noisy_labels = p_noisy_labels
        self.p_bursty = p_bursty
        self.icl_image_format = icl_image_format
        self.bursty_image_format = bursty_image_format
        self.n_support = n_support
        self.n_query = n_query
        if sum(bursty_image_format) != n_support:
            self.n_support = sum(bursty_image_format)   # if we sample bursty sequence, we need to have n_support = sum(bursty_image_format), update it

        self.transform = transforms

        # get classes based on the split
        self.get_classes(class_split_mode, split, seed, dataset_name, reduced_factor)

        self.images = []
        self.labels = []
        self.presampled_sequences = []

        self.num_classes = len(self.classes)
        self.seq_len = self.n_support*2+self.n_query
        self.bursty_image_format = bursty_image_format

        if sequences_file is not None:
            # presampled sequences provided, load them
            with open(sequences_file, "r") as f:
                    image_labels_pairs = f.readlines()
            
            for image_label_pair in image_labels_pairs:
                images, labels = image_label_pair.split("|")
                images = images.split(" ")
                labels = labels.strip("\n").split(" ")
                images = [Path(image) for image in images]
                labels = [int(label) for label in labels]
                self.presampled_sequences.append((images, labels))

            self.len = len(self.presampled_sequences)
        else:
            self.all_image_paths = []
            self.all_examples_per_class = {}
            for class_idx in self.classes.keys():
                tmp_max_index = len(self.images)
                samples_per_class = self.get_all_possible_images(self.classes[class_idx], dataset_name)
                self.images.extend(samples_per_class)
                self.labels.extend([class_idx]*len(samples_per_class))
                self.all_examples_per_class[class_idx] = list(range(tmp_max_index, tmp_max_index + len(samples_per_class)))

            if fixed_number_of_samples:
                self.len = fixed_number_of_samples
            else:
                self.len = len(self.images)

        print(f"Created dataset {dataset_name} with {len(self.images)} images and {len(self.labels)} labels from {len(self.classes)} classes with total length of {self.len} and presampled len {len(self.presampled_sequences)}. Seq Len is {self.seq_len}")

    def get_classes(self, class_split_mode, split, seed, dataset_name, reduced_factor):    
        if class_split_mode == "base_novel":
            if split == "test_icl":
                if dataset_name == "caltech":
                    self.classes = NOVEL_CLASSES_CALTECH
                elif dataset_name == "cifar":
                    self.classes = NOVEL_CLASSES_CIFAR
                elif dataset_name == "dtd":
                    self.classes = NOVEL_CLASSES_DTD
                elif dataset_name == "omniglot":
                    self.classes = NOVEL_ALPHABET_CHARACTERS
            else:
                if dataset_name == "caltech":
                    self.classes = BASE_CLASSES_CALTECH
                elif dataset_name == "cifar":
                    self.classes = BASE_CLASSES_CIFAR
                elif dataset_name == "dtd":
                    self.classes = BASE_CLASSES_DTD
                elif dataset_name == "omniglot":
                    self.classes = BASE_ALPHABET_CHARACTERS
        
        elif class_split_mode == "random":
            rng = random.Random(seed)
            if dataset_name == "caltech":
                novel_classes = sorted(rng.sample(ALL_CLASSES_CALTECH, 10))
                base_classes = sorted(list(set(list(ALL_CLASSES_CALTECH.keys())) - set(novel_classes)))
            elif dataset_name == "cifar":
                novel_classes = sorted(rng.sample(ALL_CLASSES_CIFAR, 10))
                base_classes = sorted(list(set(list(ALL_CLASSES_CIFAR.keys())) - set(novel_classes)))
            elif dataset_name == "dtd":
                novel_classes = sorted(rng.sample(ALL_CLASSES_DTD, 10))
                base_classes = sorted(list(set(list(ALL_CLASSES_DTD.keys())) - set(novel_classes)))
            elif dataset_name == "omniglot":
                novel_classes = sorted(rng.sample(ALL_ALPHABET_CHARACTERS, 23))
                base_classes = sorted(list(set(list(ALL_ALPHABET_CHARACTERS.keys())) - set(novel_classes)))

            
            print(f"Random class splitting for seed {seed}: {novel_classes}")
            self.classes = {}
            if split == "test_icl":
                classes = novel_classes
            else:
                classes = base_classes
            
            for idx, class_value in enumerate(classes):
                if dataset_name == "caltech":
                    self.classes[idx] = ALL_CLASSES_CALTECH[class_value]
                elif dataset_name == "cifar":
                    self.classes[idx] = ALL_CLASSES_CIFAR[class_value]
                elif dataset_name == "dtd":
                    self.classes[idx] = ALL_CLASSES_DTD[class_value]
                elif dataset_name == "omniglot":
                    self.classes[idx] = ALL_ALPHABET_CHARACTERS[class_value]
        else:
            raise ValueError("Split not supported!")

        if reduced_factor > 0:
            all_classes_keys = sorted(list(self.classes.keys()))
            classes_to_keep = random.sample(all_classes_keys, reduced_factor)
            new_classes = {}
            for new_idx, old_idx in enumerate(classes_to_keep):
                new_classes[new_idx] = self.classes[old_idx]
            self.classes = new_classes
        
    def get_all_possible_images(self, sampled_class, dataset_name):
        """
        Searches for all possible images for a given class (based on the sampled_class and self.classes lookup table)
        """
        if dataset_name in ("caltech", "cifar", "dtd"):
            tmp_images = sorted(list((self.root / sampled_class).glob("*.jpg")))

            if self.split == "test_icl":
                indices_to_keep = list(range(len(tmp_images)))
            else:
                all_indices = list(range(0, len(tmp_images)))
                if dataset_name in ("caltech", "dtd"):
                    test_indices = all_indices[:6]   # first 6 images are for testing
                    train_indices = all_indices[6:]
                else: # cifar
                    test_indices = list(range(0, len(tmp_images), 10))  # only take every 10th photo for validation
                    train_indices = sorted(list(set(list(range(0, len(tmp_images)))) - set(test_indices)))

                if self.split == "test_iwl":
                    indices_to_keep = test_indices
                else:
                    indices_to_keep = train_indices

            tmp_images = [tmp_img for idx, tmp_img in enumerate(tmp_images) if idx in indices_to_keep]

        else:   # omniglot
            custom_test_suffix = ["01.png", "02.png", ]
            alphabet_name, character_name = sampled_class
            if self.split == "train":
                tmp_images = sorted([p for p in (self.root / alphabet_name / character_name).glob("*.png") if p.name.split("_")[1] not in custom_test_suffix])
            elif self.split == "test_iwl":
                tmp_images = sorted([p for p in (self.root / alphabet_name / character_name).glob("*.png") if p.name.split("_")[1] in custom_test_suffix])
            else:
                tmp_images = sorted(list((self.root / alphabet_name / character_name).glob("*.png")))

        if len(tmp_images) == 0:
            raise ValueError("No images found for this class: ", sampled_class)
       
        return tmp_images

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.data_mode == "sequence": 
            if self.split == "test_iwl":
                images, labels = self.get_iwl_sequence(idx)     # will be sampled from the base classes but unseen images
            elif self.split == "test_icl":
                images, labels = self.get_icl_sequence(idx)        # will be sampled from novel classes
            else:
                if random.random() < self.p_bursty:             # if p_bursty > 0, we sample bursty sequence with probability p_bursty
                    images, labels = self.get_bursty_sequence(idx)
                else:                                           # samples normal non-bursty sequence
                    images, labels = self.get_standard_sequence(idx)
            return self.get_combined_output(images, labels)

        elif self.data_mode == "sequence_with_copy":
            if random.random() < self.p_bursty:             # if p_bursty > 0, we sample bursty sequence with probability p_bursty
                images, labels = self.get_bursty_sequence(idx, copy_paste=True)
            else:                                           # samples normal non-bursty sequence
                images, labels = self.get_standard_sequence(idx)
            return self.get_combined_output(images, labels)

        else:
            raise ValueError("Data mode not supported!")
    
    def get_combined_output(self, images_sequence, labels_sequence):
        labels_sequence = [labels_sequence[i] for i in range(len(labels_sequence))] # we need list of tensors
        combined_input_sequence = [val for pair in zip(images_sequence, labels_sequence) for val in pair]
        negatives = torch.ones(len(labels_sequence)).long() * -1
        labels_sequence_ = torch.stack(labels_sequence, dim=0)
        combined_target_sequence = torch.stack((labels_sequence_, negatives)).transpose(0,1).reshape(-1)
        return combined_input_sequence[:-1], combined_target_sequence[:-1]
    
    def get_bursty_sequence(self, idx, copy_paste=False):
        """
        Samples bursty sequence with format self.bursty_image_format
        """
        images = []
        labels = []
        if self.len == len(self.images):
            query_idx = idx
            query_label = self.labels[idx]
            query_image = self.images[idx]
        else:
            query_label = random.choice(list(self.classes.keys()))
            query_idx = random.choice(self.all_examples_per_class[query_label])
            query_image = self.images[query_idx]
            
        bursty_image_format = self.bursty_image_format
        num_classes = len(bursty_image_format)
        available_classes = sorted(list(set(self.classes.keys()) - set([query_label])))
        sampled_classes = random.sample(available_classes, num_classes)  # we sample all classes except query class
        sampled_classes[0] = query_label  # we put query class as the first one

        for sampled_class, num_samples in zip(sampled_classes, bursty_image_format):
            if sampled_class == query_label:
                available_images = list(set(self.all_examples_per_class[sampled_class]) - set([query_idx]))
                tmp_images = random.sample(available_images, num_samples)
                if copy_paste:
                    images.extend([query_idx]* num_samples)
                else:
                    images.extend(tmp_images)
            else:
                tmp_images = random.sample(self.all_examples_per_class[sampled_class], num_samples)
                if copy_paste:
                    images.extend([tmp_images[0]]* num_samples)
                else:
                    images.extend(tmp_images)
            labels.extend([sampled_class]*num_samples)
        
        indices = list(range(len(images)))
        random.shuffle(indices)
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]

        images = [self.images[tmp_idx] for tmp_idx in images]  # convert indices to actual image paths

        # add noise
        if self.p_noisy_labels > 0:
            if random.random() < self.p_noisy_labels:
                old_query_label = query_label
                query_label = random.choice(list(self.classes.keys()))
                for idx, label in enumerate(labels):
                    if label == old_query_label:
                        labels[idx] = query_label

        images.append(query_image)
        labels.append(query_label)

        images = [self.transform((Image.open(image)).convert("RGB")) for image in images]
        return torch.stack(images), torch.tensor(np.array(labels))
        
    def get_standard_sequence(self, idx):
        """
        Samples non-bursty sequence
        """
        images, labels = [], []
 
        # sample query item -- if len > dataset length, we sample randomly
        if self.len == len(self.images):
            query_label = self.labels[idx]
            query_image = self.images[idx]
        else:
            query_label = random.choice(list(self.classes.keys()))
            query_image = self.images[random.choice(self.all_examples_per_class[query_label])]
 
        available_classes = sorted(list(set(self.classes.keys()) - set([query_label])))
        support_classes = random.sample(available_classes, self.n_support)

        # get support images
        for sampled_class in support_classes:
            tmp_image = random.choice(self.all_examples_per_class[sampled_class])
            images.append(tmp_image)
            labels.append(sampled_class)

        # add noise
        if self.p_noisy_labels > 0 and random.random() < self.p_noisy_labels:
            query_label = random.choice(list(self.classes.keys()))

        
        images = [self.images[tmp_idx] for tmp_idx in images]  # convert indices to actual image paths

        images.append(query_image)
        labels.append(query_label)

        images = [self.transform((Image.open(image)).convert("RGB")) for image in images]
        return torch.stack(images), torch.tensor(np.array(labels))
         
    def get_iwl_sequence(self, idx):
        """
        Samples iwl sequence - we sample from base classes but the images are unseen
        """
        images, labels = [], []

        if self.len == len(self.images):
            query_label = self.labels[idx]
            query_image = self.images[idx]
        else:
            query_label = random.choice(list(self.classes.keys()))
            query_image = self.images[random.choice(self.all_examples_per_class[query_label])]

        support_classes = random.sample(sorted(list(set(self.classes.keys()) - set([query_label]))), self.n_support)

        # get support images
        for sampled_class in support_classes:
            tmp_image = random.choice(self.all_examples_per_class[sampled_class])
            images.append(tmp_image)
            labels.append(sampled_class)

        images = [self.images[tmp_idx] for tmp_idx in images]  # convert indices to actual image paths
        
        # get query image
        images.append(query_image)
        labels.append(query_label)


        images = [self.transform((Image.open(image)).convert("RGB")) for image in images]
        return torch.stack(images), torch.tensor(np.array(labels))
    
    def get_icl_sequence(self, idx):
        """
        Samples icl sequence - we sample from novel classes. 
        The format of a sequence is determined by self.icl_image_format  2way4shot, 4way2shot, etc.
        """

        if self.presampled_sequences:
            images, labels = self.presampled_sequences[idx]
            images = [self.transform((Image.open(image)).convert("RGB")) for image in images]
            return torch.stack(images), torch.tensor(np.array(labels))
        else:
            images = []
            labels = []
            query_idx = random.choice(range(len(self.icl_image_format)))
            query_image = None
            num_classes = len(self.icl_image_format)
            sampled_classes = random.sample(list(self.classes.keys()), num_classes)

            for class_idx, sampled_class in enumerate(sampled_classes):
                num_samples = self.icl_image_format[class_idx]
                
                if class_idx == query_idx:
                    num_samples += 1
                
                tmp_images = random.sample(self.all_examples_per_class[sampled_class], num_samples)
                images.extend(tmp_images)
                labels.extend([class_idx]*num_samples)
                
                if class_idx == query_idx:
                    query_image = images[-1]
                    del images[-1]
                    del labels[-1]
            
            # permute support images and create sequence
            indices = list(range(len(images)))
            random.shuffle(indices)
            images = [images[i] for i in indices]
            labels = [labels[i] for i in indices]

            images.append(query_image)
            labels.append(query_idx)

            images = [self.transform((Image.open(self.images[image_idx])).convert("RGB")) for image_idx in images]
            return torch.stack(images), torch.tensor(np.array(labels))


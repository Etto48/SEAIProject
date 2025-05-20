import torch
import torch.random as random
from torchvision import datasets, transforms
from torch.utils.data import IterableDataset
import numpy as np

class SplitCIFAR10(IterableDataset):
    def __init__(self, classes_per_split=2, task_duration=1000, root="./data", download=True, transform=None, train=True):
        self.root = root
        self.transform = transforms.ToTensor() if transform is None else transform
        self.download = download
        self.train = train
        self.dataset = datasets.CIFAR10(root=self.root, train=train, download=self.download)
        self.classes_per_split = classes_per_split
        self.buckets = self.create_buckets()
        self.task_duration = task_duration

    def create_buckets(self):
        # Create buckets of samples with classes_per_split classes in each bucket
        targets = np.array(self.dataset.targets)
        classes = np.unique(targets)
        # check if classes_per_split is a divisor of the number of classes
        if len(classes) % self.classes_per_split != 0:
            raise ValueError(f"Number of classes {len(classes)} is not divisible by classes_per_split {self.classes_per_split}")
        num_buckets = len(classes) // self.classes_per_split
        buckets = []
        for i in range(num_buckets):
            start = i * self.classes_per_split
            end = start + self.classes_per_split
            bucket_classes = classes[start:end]
            bucket_indices = np.isin(targets, bucket_classes)
            buckets.append(np.where(bucket_indices)[0])
        return buckets
    
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.buckets) * self.task_duration

    def __iter__(self):
        for current_bucket in range(len(self.buckets)):
            for _ in range(self.task_duration):
                bucket = self.buckets[current_bucket]
                idx = np.random.choice(bucket)
                img, target = self.dataset[idx]
                img = self.transform(img)
                target = torch.tensor(target, dtype=torch.long)
                task_id = torch.tensor(current_bucket, dtype=torch.long)
                yield img, target, task_id

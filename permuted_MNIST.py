import torch
import torch.random as random
from torchvision import datasets, transforms
from torch.utils.data import IterableDataset
import numpy as np

class PermutedMNIST(IterableDataset):
    def __init__(self, num_permutations=10, transition_steps=1000, root="./data", download=True):
        self.root = root
        self.transform = transforms.ToTensor()
        self.download = download
        self.dataset = datasets.MNIST(root=self.root, train=True, download=self.download)
        self.permutation_id = 0
        self.transition = 0
        self.transition_steps = transition_steps
        self.num_permutations = num_permutations

    def _get_permutation(self, permutation_id: int):
        # this resets the random state when the context is closed, so that the next call to random will not be affected
        with random.fork_rng():
            # seed the permutation
            torch.manual_seed(permutation_id + 0xdeadbeef)
            perm = torch.randperm(28 * 28)
        return perm
    
    def __iter__(self):
        while True:
            next_id = (self.permutation_id + 1) % self.num_permutations
            # simulate a gradual transition
            permutation_id = np.random.choice([self.permutation_id, next_id], p=[1-self.transition, self.transition])
            self.transition += 1.0 / self.transition_steps
            if self.transition >= 1.0:
                self.transition = 0
                self.permutation_id = next_id
            perm = self._get_permutation(permutation_id)
            img_index = np.random.randint(0, len(self.dataset))
            img, target = self.dataset[img_index]
            img = self.transform(img)
            img = img.view(-1)
            img = img[perm]
            img = img.view(28, 28)
            target = torch.tensor(target, dtype=torch.long)
            permutation_id = torch.tensor(permutation_id, dtype=torch.long)
            yield img, target, permutation_id
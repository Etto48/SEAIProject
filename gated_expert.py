import copy
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

from expert import ExpertMLP
from gate import GateAutoencoder
from split_MNIST import SplitMNIST
from split_CIFAR10 import SplitCIFAR10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class GatedExpert(nn.Module):
    def __init__(self, in_out_shape=(3, 32, 32), classes=10, depth=3, ff_depth=3, expert_depth=3, hidden_dim=128, latent_dim=128, task_aware=True):
        super(GatedExpert, self).__init__()
        self.gates = nn.ModuleList()
        self.experts = nn.ModuleList()
        self.gate_optimizers = []
        self.expert_optimizers = []
        self.in_out_shape = in_out_shape
        self.classes = classes
        self.depth = depth
        self.ff_depth = ff_depth
        self.expert_depth = expert_depth
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hold_time_after_new_task = 10
        self.time_since_new_task = 0
        self.temperature = 2.0
        self.error_threshold = 0.25
        self.selection_softmax = nn.Softmax(dim=0)
        self.gate_loss = nn.L1Loss(reduction='none')
        self.expert_loss = nn.CrossEntropyLoss()
        self.task_aware = task_aware

    def new_task(self):
        self.time_since_new_task = 0
        gate = GateAutoencoder(
            in_out_shape=self.in_out_shape,
            depth=self.depth,
            ff_depth=self.ff_depth,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
        self.gates.append(gate)
        expert = ExpertMLP(
            input_feature=self.latent_dim,
            depth=self.expert_depth,
            hidden_features=self.hidden_dim,
            output_features=self.classes
        )
        self.experts.append(expert)
        self.gate_optimizers.append(torch.optim.Adam(gate.parameters(), lr=1e-3))
        self.expert_optimizers.append(torch.optim.Adam(expert.parameters(), lr=1e-3))
    
    def new_task_was_recently_added(self):
        return self.time_since_new_task < self.hold_time_after_new_task and not self.task_aware

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x (B, C, H, W)
        latent_representations = []
        reconstructions = []
        reconstruction_errors = []
        for gate in self.gates:
            recon, latent = gate(x)
            latent_representations.append(latent)
            reconstructions.append(recon)
            error = self.gate_loss(recon, x)
            error = error.mean(dim=(1, 2, 3))
            reconstruction_errors.append(error)

        # reconstruction_errors (N_gates, B)
        reconstruction_errors = torch.stack(reconstruction_errors, dim=0)
        reconstructions = torch.stack(reconstructions, dim=0)
        # latent_representations (N_gates, B, latent_dim)
        latent_representations = torch.stack(latent_representations, dim=0)
        relevance_logits = -reconstruction_errors/self.temperature
        if self.new_task_was_recently_added():
            relevance_logits[-1, :] /= self.temperature
        # relevance_scores (N_gates, B)
        relevance_scores = self.selection_softmax(relevance_logits)
        # indices (B)
        min_reconstruction_errors, indices = torch.min(reconstruction_errors, dim=0)
        # mask (N_gates, B)
        if mask is None:
            mask = torch.arange(len(self.gates)).unsqueeze(1) == indices.unsqueeze(0)
        # classes (B, classes)
        logits = torch.zeros(x.shape[0], self.classes)
        for i, expert in enumerate(self.experts):
            if torch.all(~mask[i]):
                continue
            expert_input = latent_representations[i][mask[i]]
            expert_output = expert(expert_input)
            logits[mask[i]] = expert_output

        return logits, reconstructions, indices, min_reconstruction_errors, relevance_scores, mask

    def mask_from_task_ids(self, task_ids: torch.Tensor):
        max_task_id = max(task_ids.max(), len(self.gates) - 1)
        mask = torch.arange(max_task_id + 1).unsqueeze(1) == task_ids.unsqueeze(0)
        return mask

    def fit(self, train_dataset: IterableDataset, test_dataset: Dataset | None = None):
        train_loader = DataLoader(train_dataset, batch_size=32)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=32, generator=torch.Generator(device=device))

        loading_bar = tqdm(train_loader, total=len(train_loader), desc="Training", unit="batch")
        for i, batch in enumerate(loading_bar):
            images, targets, task_ids = batch
            images = images.to(device)
            targets = targets.to(device)
            task_ids = task_ids.to(device)
            if task_ids.max() > len(self.gates) - 1:
                self.new_task()

            self.train()
            mask = self.mask_from_task_ids(task_ids)
            for j in range(len(self.gates)):
                if torch.all(~mask[j]):
                    continue
                self.gate_optimizers[j].zero_grad()
                self.expert_optimizers[j].zero_grad()
                recon, latent = self.gates[j](images[mask[j]])
                expert_output = self.experts[j](latent.detach())
                gate_loss = self.gate_loss(recon, images[mask[j]]).mean()
                expert_loss = self.expert_loss(expert_output, targets[mask[j]])
                gate_loss.backward()
                expert_loss.backward()
                self.gate_optimizers[j].step()
                self.expert_optimizers[j].step()
                
            loading_bar.set_postfix({
                #"loss": f"{loss.item():.3f}", 
                "e": f"{expert_loss.item():.3f}", 
                "g": f"{gate_loss.item():.3f}", 
                "n": len(self.gates)})
        if test_dataset is not None:
            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                loading_bar = tqdm(test_loader, desc="Testing", total=len(test_loader))
                confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
                for x, y in loading_bar:
                    x = x.to(device)
                    y = y.to(device)
                    logits, reconstructions, indices, min_reconstruction_errors, relevance_scores, mask = self(x)
                    outputs = logits
                    total += y.shape[0]
                    correct += (outputs.argmax(dim=1) == y).sum().item()
                    accuracy = correct / total
                    confusion_matrix += torch.bincount(y * 10 + outputs.argmax(dim=1), minlength=100).reshape(10, 10)
                    loading_bar.set_postfix(accuracy=f"{accuracy:.2%}")
            
            plt.figure(figsize=(10, 10))
            plt.imshow(confusion_matrix.cpu(), interpolation='nearest')
            plt.colorbar()
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.xticks(np.arange(10))
            plt.yticks(np.arange(10))
            plt.show()
            # demo some images and autoencode them with all gates
            original_images = []
            while len(original_images) < 10:
                for i in range(len(test_dataset)):
                    img, label = test_dataset[i]
                    if label == len(original_images):
                        original_images.append((img, label))
            original_images = torch.stack([img for img, _ in original_images], dim=0).to(device)
            reconstructions = []
            for gate in self.gates:
                recon, _ = gate(original_images)
                reconstructions.append(recon)
            reconstructions = torch.stack(reconstructions, dim=0)
            original_images = original_images.detach().cpu().numpy()
            reconstructions = reconstructions.detach().cpu().numpy()
            fig, axes = plt.subplots(len(self.gates)+1, 10)
            for i in range(10):
                axes[0, i].imshow(original_images[i].transpose(1, 2, 0), cmap='gray')
                axes[0, i].axis('off')
                for j in range(len(self.gates)):
                    axes[j + 1, i].imshow(reconstructions[j, i].transpose(1, 2, 0), cmap='gray')
                    axes[j + 1, i].axis('off')
            plt.show()

def main():
    train_dataset = SplitCIFAR10(task_duration=10000)
    #test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=train_dataset.transform, target_transform=torch.tensor)
    test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=train_dataset.transform)
    model = GatedExpert()
    model.fit(train_dataset, test_dataset)

    
if __name__ == "__main__":
    main()

            

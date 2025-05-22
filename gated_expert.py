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
    def __init__(self, in_out_shape=(1, 28, 28), classes=10, depth=2, ff_depth=3, expert_depth=3, hidden_dim=64, latent_dim=128):
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
        self.selection_softmax = nn.Softmax(dim=0)
        self.gate_loss = nn.L1Loss(reduction='none')
        self.expert_loss = nn.CrossEntropyLoss()
        self.new_task_boost = 100

        self.error_window = []
        self.error_window_size = 64
        self.error_window_sum = 0

        self.error_std_threshold = 4
        self.error_flat_threshold = 0.01

        self.fine_tune = False

        self.new_task()

    def add_error(self, error) -> tuple[float, float]:
        if len(self.error_window) >= self.error_window_size:
            self.error_window_sum -= self.error_window.pop(0)
        self.error_window.append(error)
        self.error_window_sum += error
        mean = self.error_window_sum / len(self.error_window)
        std = (sum((error - mean) ** 2 for error in self.error_window) / len(self.error_window)) ** 0.5
        return mean, std

    def reset_error_window(self):
        self.error_window = []
        self.error_window_sum = 0

    def new_task(self):
        self.time_since_new_task = 0
        gate = copy.deepcopy(self.gates[-1]) if len(self.gates) > 0 and self.fine_tune else GateAutoencoder(
            in_out_shape=self.in_out_shape,
            depth=self.depth,
            ff_depth=self.ff_depth,
            conv_hidden_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 16,
            latent_dim=self.latent_dim
        )
        self.gates.append(gate)
        expert = copy.deepcopy(self.experts[-1]) if len(self.experts) > 0 and self.fine_tune else ExpertMLP(
            input_feature=self.latent_dim,
            depth=self.expert_depth,
            hidden_features=self.hidden_dim,
            output_features=self.classes
        )
        self.experts.append(expert)
        self.gate_optimizers.append(torch.optim.Adam(gate.parameters(), lr=1e-3))
        self.expert_optimizers.append(torch.optim.Adam(expert.parameters(), lr=1e-3))
        self.reset_error_window()
    
    def forward_gates(self, x: torch.Tensor):
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
        return reconstructions, latent_representations, reconstruction_errors

    def forward(self, x: torch.Tensor, mask: torch.Tensor, latent_representations: torch.Tensor):
        # classes (B, classes)
        logits = torch.zeros(x.shape[0], self.classes)
        for i, expert in enumerate(self.experts):
            if torch.all(~mask[i]):
                continue
            expert_input = latent_representations[i][mask[i]]
            expert_output = expert(expert_input)
            logits[mask[i]] = expert_output

        return logits

    def mask_from_recon_errors(self, reconstruction_errors: torch.Tensor):
        last_task_boost = torch.zeros_like(reconstruction_errors)
        if self.training:
            last_task_boost[-1] = self.new_task_boost
        min_reconstruction_errors, indices = torch.min(reconstruction_errors - last_task_boost, dim=0)
        mask = torch.arange(len(self.gates)).unsqueeze(1) == indices.unsqueeze(0)
        return mask, min_reconstruction_errors

    def mask_from_task_ids(self, task_ids: torch.Tensor):
        max_task_id = max(task_ids.max(), len(self.gates) - 1)
        mask = torch.arange(max_task_id + 1).unsqueeze(1) == task_ids.unsqueeze(0)
        return mask

    def predict(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            reconstructions, latent_representations, reconstruction_errors = self.forward_gates(x)
            mask, _ = self.mask_from_recon_errors(reconstruction_errors)
            logits = self.forward(x, mask, latent_representations)
            return logits, reconstructions

    def fit(self, train_dataset: IterableDataset, test_dataset: Dataset | None = None):
        train_loader = DataLoader(train_dataset, batch_size=16)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=64, generator=torch.Generator(device=device))

        loading_bar = tqdm(train_loader, total=len(train_loader), desc="Training", unit="batch")
        mean, std = 0, 0
        accuracy = 0
        for i, batch in enumerate(loading_bar):
            images, targets, task_ids = batch
            images = images.to(device)
            targets = targets.to(device)

            reconstructions, latent_representations, reconstruction_errors = self.forward_gates(images)
            mask, _ = self.mask_from_recon_errors(reconstruction_errors)
            avg_selected_recon_errors = reconstruction_errors[mask].mean().item()

            if len(self.error_window) == self.error_window_size and avg_selected_recon_errors > mean + self.error_std_threshold * std + self.error_flat_threshold:
                self.new_task()
                reconstructions, latent_representations, reconstruction_errors = self.forward_gates(images)
                mask, _ = self.mask_from_recon_errors(reconstruction_errors)
                avg_selected_recon_errors = reconstruction_errors[mask].mean().item()

            mean, std = self.add_error(avg_selected_recon_errors)

            self.train()
            for j in range(len(self.gates)):
                if mask[j].sum() < 2: # exclude empty batches and single samples to avoid BatchNorm errors
                    continue
                self.gate_optimizers[j].zero_grad()
                self.expert_optimizers[j].zero_grad()
                images_j = images[mask[j]]
                targets_j = targets[mask[j]]
                recon, latent = self.gates[j](images_j)
                expert_output = self.experts[j](latent.detach())
                correct = (expert_output.argmax(dim=1) == targets_j).sum().item()
                total = targets_j.shape[0]
                accuracy = (correct / total) * 0.1 + accuracy * 0.9
                gate_loss = self.gate_loss(recon, images_j).mean()
                expert_loss = self.expert_loss(expert_output, targets_j)
                gate_loss.backward()
                expert_loss.backward()
                self.gate_optimizers[j].step()
                self.expert_optimizers[j].step()
                
            loading_bar.set_postfix({
                #"loss": f"{loss.item():.3f}", 
                "e": f"{expert_loss.item():.3f}", 
                "g": f"{gate_loss.item():.3f}", 
                "n": len(self.gates),
                "mean": f"{mean:.3f}",
                "std": f"{std:.3f}",
                "acc": f"{accuracy:.2%}"})
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
                    logits, reconstructions = self.predict(x)
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
    train_dataset = SplitMNIST(task_duration=10000)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=train_dataset.transform)
    model = GatedExpert()
    model.fit(train_dataset, test_dataset)

    
if __name__ == "__main__":
    main()

            

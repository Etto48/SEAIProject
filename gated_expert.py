import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm

from expert import ExpertMLP
from gate import GateAutoencoder
from split_CIFAR10 import SplitCIFAR10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class GatedExpert(nn.Module):
    def __init__(self, in_out_shape=(3, 32, 32), classes=10, depth=3, hidden_dim=128, latent_dim=128):
        super(GatedExpert, self).__init__()
        self.gates = nn.ModuleList()
        self.experts = nn.ModuleList()
        self.gate_optimizers = []
        self.expert_optimizers = []
        self.in_out_shape = in_out_shape
        self.classes = classes
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hold_time_after_new_task = 4
        self.time_since_new_task = self.hold_time_after_new_task + 1
        self.temperature = 2.0
        self.error_threshold = 0.5
        self.selection_softmax = nn.Softmax(dim=0)
        self.gate_loss = nn.L1Loss()
        self.expert_loss = nn.CrossEntropyLoss()
        self.new_task()

    def new_task(self):
        self.time_since_new_task = 0
        gate = GateAutoencoder(
            in_out_shape=self.in_out_shape,
            depth=self.depth,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
        self.gates.append(gate)
        expert = ExpertMLP(
            input_feature=self.latent_dim,
            hidden_features=self.hidden_dim,
            output_features=self.classes
        )
        self.experts.append(expert)
        self.gate_optimizers.append(torch.optim.Adam(gate.parameters(), lr=1e-3))
        self.expert_optimizers.append(torch.optim.Adam(expert.parameters(), lr=1e-3))
    
    def new_task_was_recently_added(self):
        return self.time_since_new_task < self.hold_time_after_new_task

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x (B, C, H, W)
        latent_representations = []
        reconstruction_errors = []
        for gate in self.gates:
            recon, latent = gate(x)
            latent_representations.append(latent)
            error = self.gate_loss(recon, x)
            reconstruction_errors.append(error)

        # reconstruction_errors (N_gates, B)
        reconstruction_errors = torch.stack(reconstruction_errors, dim=0)
        # latent_representations (N_gates, B, latent_dim)
        latent_representations = torch.stack(latent_representations, dim=0)
        relevance_logits = -reconstruction_errors/self.temperature
        if self.new_task_was_recently_added():
            relevance_logits[-1, :] /= self.temperature
        # relevance_scores (N_gates, B)
        relevance_scores = self.selection_softmax()
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

        return logits, indices, min_reconstruction_errors, relevance_scores, mask

    def fit(self, train_dataset: IterableDataset, test_dataset: IterableDataset | None = None):
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        loading_bar = tqdm(total=len(train_loader), desc="Training", unit="batch")
        for i, batch in enumerate(loading_bar):
            images, targets, task_ids = batch
            images = images.to(device)
            targets = targets.to(device)

            self.train()
            self.zero_grad()
            logits, indices, min_reconstruction_errors, relevance_scores, mask = self(images)
            if torch.any(min_reconstruction_errors > self.error_threshold) and \
                not self.new_task_was_recently_added():
                self.new_task()
                mask[indices][min_reconstruction_errors > self.error_threshold] = False
                mask = torch.cat([mask, torch.zeros((images.shape[0], 1), dtype=torch.bool)])
                mask[-1][min_reconstruction_errors > self.error_threshold] = True
                logits, indices, min_reconstruction_errors, relevance_scores, mask = self(images, mask)
            else:
                self.time_since_new_task += 1
            expert_loss = self.expert_loss(logits, targets)
            gate_loss = min_reconstruction_errors.mean()
            loss = expert_loss + gate_loss
            loss.backward()
            for i, (gate_optimizer, expert_optimzer) \
                in enumerate(zip(self.gate_optimizers, self.expert_optimizers)):
                if torch.any(mask[i]):
                    gate_optimizer.step()
                    expert_optimzer.step()
            loading_bar.set_postfix({"loss": loss.item(), "e": expert_loss.item(), "g": gate_loss.item()})


def main():
    dataset = SplitCIFAR10()
    model = GatedExpert()
    model.fit(dataset)
    
if __name__ == "__main__":
    main()

            

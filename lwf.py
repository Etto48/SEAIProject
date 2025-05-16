import torch
import torch.nn as nn
import torchvision
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class LWFClassifier(nn.Module):
    def __init__(self, in_out_shape=(3, 32, 32), classes=10):
        super(LWFClassifier, self).__init__()
        self.feature_extractor = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        self.classifier_input_dim = self.feature_extractor.classifier[6].in_features
        self.feature_extractor.classifier.pop(6)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.old_classifier_head: nn.Linear | None = None
        self.classes = classes
        self.classifier_head = nn.Linear(self.classifier_input_dim, classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss = nn.CrossEntropyLoss()

        self.error_window_max_len = 10
        self.error_threshold = 3
        self.error_window = []
        self.error_window_sum = 0

        self.temperature = 2
        self.old_loss_weight = 1

    def new_error(self, error: float) -> tuple[float, float]:
        if len(self.error_window) >= self.error_window_max_len:
            self.error_window_sum -= self.error_window.pop(0)
        self.error_window.append(error)
        self.error_window_sum += error

        mean = self.error_window_sum / len(self.error_window)
        std = (sum((x - mean) ** 2 for x in self.error_window) / len(self.error_window)) ** 0.5
        return mean, std
    
    def new_task(self, classes: int):
        self.old_classifier_head = self.classifier_head
        for param in self.old_classifier_head.parameters():
            param.requires_grad = False
        self.classes = classes
        self.classifier_head = nn.Linear(self.classifier_input_dim, classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        if self.old_classifier_head is not None:
            x_old = self.old_classifier_head(x)
        else:
            x_old = None
        x = self.classifier_head(x)
        return x, x_old

    def fit(self, train_dataset: IterableDataset, test_dataset: Dataset | None = None):
        train_loader = DataLoader(train_dataset, batch_size=32)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        loading_bar = tqdm(train_loader, desc="Training", unit="batch")
        mean, std = 0, 0
        for batch in loading_bar:
            img, label, task = batch
            img = img.to(device)
            label = label.to(device)
            task = task.to(device)

            self.optimizer.zero_grad()
            output, old_output = self(img)
            if old_output is not None:
                old_output /= self.temperature
            
            loss_new: torch.Tensor = self.loss(output, label)
            if std > 0 and loss_new.item() > mean + self.error_threshold * std:
                self.new_task(self.classes)
            mean, std = self.new_error(loss_new.item())
            

            loss_old = torch.zeros_like(loss_new)
            if old_output is not None:
                loss_old = self.loss(old_output, label)
                loss_old = self.old_loss_weight * loss_old
            loss: torch.Tensor = loss_new + loss_old
            loss.backward()
            self.optimizer.step()
            loading_bar.set_postfix({"loss": loss.item(), "loss_new": loss_new.item(), "loss_old": loss_old.item()})

        


if __name__ == "__main__":
    model = LWFClassifier()
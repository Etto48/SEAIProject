import copy
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm.auto import tqdm
from split_CIFAR10 import SplitCIFAR10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int ,out_dim: int):
        super(ClassificationHead, self).__init__()
        self.seq = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.seq(x)
        return x

class LWFClassifier(nn.Module):
    def __init__(self, in_out_shape=(3, 32, 32), classes=10):
        super(LWFClassifier, self).__init__()
        self.feature_extractor = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        self.classifier_input_dim = self.feature_extractor.classifier[1].in_features
        self.classifier_hidden_dim = self.feature_extractor.classifier[1].out_features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.old_classifier_heads: nn.ModuleList[ClassificationHead] = nn.ModuleList()
        self.classes = classes
        self.lr = 1e-3
        self.classifier_head = ClassificationHead(
            self.classifier_input_dim, 
            self.classifier_hidden_dim,
            classes)
        self.optimizer = torch.optim.SGD(self.classifier_head.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        self.loss_old = lambda logx, logy: -torch.sum(torch.softmax(logy, dim=1) * torch.log_softmax(logx, dim=1), dim=1).mean()

        self.error_window_max_len = 32
        self.error_threshold = 10
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
        self.old_classifier_heads.append(self.classifier_head)
        self.classifier_head = copy.deepcopy(self.classifier_head)
        for param in self.old_classifier_heads[-1].parameters():
            param.requires_grad = False
        self.old_classifier_heads[-1].eval()
        self.classes = classes
        self.optimizer = torch.optim.SGD(self.classifier_head.parameters(), lr=self.lr)
        self.error_window = []
        self.error_window_sum = 0
        self.batches_with_high_loss = 0

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor.features(x)
        x = self.feature_extractor.avgpool(x)
        x = torch.flatten(x, 1)
        x_old = []
        for i in range(len(self.old_classifier_heads)):
            x_old.append(self.old_classifier_heads[i](x))

        x = self.classifier_head(x)
        return x, x_old

    def fit(self, train_dataset: IterableDataset, test_dataset: Dataset | None = None):
        train_loader = DataLoader(train_dataset, batch_size=32)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        loading_bar = tqdm(train_loader, desc="Training", unit="batch")
        mean, std = 0, 0
        task_changes = 0
        for batch in loading_bar:
            img, label, task = batch
            img = img.to(device)
            label = label.to(device)
            task = task.to(device)

            self.optimizer.zero_grad()
            output, old_output = self(img)
            
            loss_new: torch.Tensor = self.loss(output, label)
            if len(self.error_window) == self.error_window_max_len and loss_new.item() > mean + self.error_threshold * std:
                self.new_task(self.classes)
                task_changes += 1
            mean, std = self.new_error(loss_new.item())
            

            loss_old = torch.zeros_like(loss_new)
            for i in range(len(old_output)):
                loss_old += self.old_loss_weight * self.loss_old(output/self.temperature, old_output[i]/self.temperature)
            loss: torch.Tensor = loss_new + loss_old
            loss.backward()
            self.optimizer.step()
            loading_bar.set_postfix({
                "loss": loss.item(), 
                "loss_new": loss_new.item(), 
                "loss_old": loss_old.item(), 
                "mean": mean, 
                "std": std,
                "n": task_changes})
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
                    logits, _ = self(x)
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
        

def main():
    cifar_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])
    train_dataset = SplitCIFAR10(task_duration=10000, transform=cifar_transform)
    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=train_dataset.transform)
    model = LWFClassifier()
    model.fit(train_dataset, test_dataset)

if __name__ == "__main__":
    main()
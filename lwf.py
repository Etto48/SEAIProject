import torch
import torch.nn as nn
import torchvision
from torch.utils.data import IterableDataset, Dataset, DataLoader

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

    def new_error(self, error: float):
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
        x = self.classifier_head(x)
        return x

    def fit(self, train_dataset: IterableDataset, test_dataset: Dataset | None = None):
        train_loader = DataLoader(train_dataset, batch_size=32)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        


if __name__ == "__main__":
    model = LWFClassifier()
import copy
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm.auto import tqdm
from split_CIFAR10 import SplitCIFAR10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Helper functions for WideResNet
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """Basic Block for WideResNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class WideResNet(nn.Module):
    """
    Wide Residual Network (WRN) model.
    This implementation is based on the WRN-20-8 architecture.
    - Depth: 20 (3 stages of 3 BasicBlocks each, plus initial conv and final pooling)
    - Width factor: 8
    """
    def __init__(self, in_channels: int, width_factor: int = 8, 
                 layers: list[int] = None, zero_init_residual: bool = True):
        super(WideResNet, self).__init__()
        
        if layers is None:
            layers = [3, 3, 3] # Corresponds to WRN-20 (3 blocks per stage for 3 stages)

        block = BasicBlock # Using BasicBlock for WRN
        self.inplanes = 16 # Initial number of planes

        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks (3 stages)
        self.layer1 = self._make_layer(block, 16 * width_factor, layers[0])
        self.layer2 = self._make_layer(block, 32 * width_factor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width_factor, layers[2], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Output feature dimension will be 64 * width_factor * block.expansion
        self.feature_dim = 64 * width_factor * block.expansion

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # Downsample if stride is not 1 or if inplanes do not match planes * expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion # Update inplanes for the next block/layer
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Pooling and flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # Flatten the features
        return x

# Placeholder for ClassificationHead if not defined elsewhere
# class ClassificationHead(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes):
#         super(ClassificationHead, self).__init__()
#         # Example: a 2-layer MLP head
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, num_classes)
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)
# If ClassificationHead is just nn.Linear(input_dim, num_classes), 
# then LWFClassifier's use of hidden_dim needs adjustment.
# Assuming ClassificationHead(input_dim, hidden_dim, classes) is the intended signature.

class LWFClassifier(nn.Module):
    def __init__(self, in_out_shape=(3, 32, 32), total_classes=10, num_classes_per_task=2):
        super(LWFClassifier, self).__init__()
        
        self.feature_extractor = WideResNet(in_channels=in_out_shape[0], width_factor=8)
        self.classifier_input_dim = self.feature_extractor.feature_dim 
        self.classifier_hidden_dim = 1000

        # DO NOT FREEZE the feature extractor if it's trained from scratch
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False 
        
        self.old_classifier_heads: nn.ModuleList = nn.ModuleList()
        
        self.total_classes = total_classes
        self.num_classes_per_task = num_classes_per_task
        self.classes = self.total_classes 

        self.lr = 0.01 # Increased learning rate for training from scratch
        self.classifier_head = ClassificationHead(
            self.classifier_input_dim, 
            self.classifier_hidden_dim,
            self.total_classes 
        )
        
        # Optimizer now includes parameters from both feature_extractor and classifier_head
        self.optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier_head.parameters()), 
            lr=self.lr
        )
        self.loss = nn.CrossEntropyLoss()
        self.loss_old = lambda logx, logy: -torch.sum(torch.softmax(logy, dim=1) * torch.log_softmax(logx, dim=1), dim=1).mean()

        # Parameters for task change detection (anomaly detection)
        self.error_window_max_len = 32
        self.error_threshold = 10 
        self.error_window = []
        self.error_window_sum = 0

        self.temperature = 2
        self.old_loss_weight = 1
        self.current_task_id_inferred = 0 # Model's internal belief of current task, for capping new_task calls

    def new_error(self, error: float) -> tuple[float, float]:
        if len(self.error_window) >= self.error_window_max_len:
            self.error_window_sum -= self.error_window.pop(0)
        self.error_window.append(error)
        self.error_window_sum += error

        mean = self.error_window_sum / len(self.error_window)
        std = (sum((x - mean) ** 2 for x in self.error_window) / len(self.error_window)) ** 0.5
        return mean, std
    
    # Called when loss anomaly detected. Head structure does not change (always total_classes).
    def new_task(self): 
        print(f"\\\\nLoss-based anomaly detected. Saving current {self.total_classes}-class head and deepcopying for continued training.")
        self.current_task_id_inferred +=1 # Increment model's internal inferred task counter
        
        self.old_classifier_heads.append(self.classifier_head) # Save the current head
        # New head starts with the weights of the old one (LwF principle)
        self.classifier_head = copy.deepcopy(self.classifier_head) 
        
        # Freeze parameters of the saved old head
        for param in self.old_classifier_heads[-1].parameters():
            param.requires_grad = False
        self.old_classifier_heads[-1].eval()
        
        # Ensure the new head is trainable (it should be by default after deepcopy)
        for param in self.classifier_head.parameters():
            param.requires_grad = True
        
        # Re-initialize optimizer with feature_extractor parameters and the NEW classifier_head parameters
        self.optimizer = torch.optim.SGD(
            list(self.feature_extractor.parameters()) + list(self.classifier_head.parameters()), 
            lr=self.lr
        )
        self.error_window = []
        self.error_window_sum = 0

    def forward(self, x: torch.Tensor):
        # Extract features using the new WideResNet
        features = self.feature_extractor(x)
        
        x_old = []
        for i in range(len(self.old_classifier_heads)):
            # Old heads are also total_classes-dimensional
            x_old.append(self.old_classifier_heads[i](features))

        # Current head is total_classes-dimensional
        x_current = self.classifier_head(features)
        return x_current, x_old

    def fit(self, train_dataset: IterableDataset, test_dataset: Dataset | None = None):
        train_loader = DataLoader(train_dataset, batch_size=32)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        loading_bar = tqdm(train_loader, desc="Training", unit="batch")
        mean, std = 0, 0
        # task_changes counts how many times new_task is called (inferred task changes)
        # This is self.current_task_id_inferred now, starting from 0 for the first task.
        # So, number of *changes* will be self.current_task_id_inferred.

        for batch_idx, batch in enumerate(loading_bar):
            # original_label are global labels (e.g., 0-9 for CIFAR-10)
            # task_id_tensor indicates the task segment from SplitCIFAR10 (e.g., 0, 1, 2, 3, 4)
            img, original_label, task_id_tensor = batch 
            img = img.to(device)
            original_label = original_label.to(device) # Use global labels directly
            # task_id_tensor = task_id_tensor.to(device) # Not used for loss or head switching

            self.optimizer.zero_grad()
            # output is from the current total_classes-dimensional head
            output, old_outputs_from_model = self(img) 
            
            # Loss for the current task, using global original_label
            loss_new: torch.Tensor = self.loss(output, original_label)
            current_loss_val = loss_new.item()

            # --- Loss-based Anomaly Detection for Task Change ---
            # Max inferred changes: (total_dataset_classes / num_classes_per_task_in_split) - 1
            # E.g. CIFAR-10 (10 classes) split into 5 tasks of 2 classes each: (10/2)-1 = 4 changes.
            # self.current_task_id_inferred tracks the number of tasks seen (0 for 1st, 1 for 2nd, etc.)
            # So, number of changes = self.current_task_id_inferred.
            max_inferred_task_segments = (self.total_classes // self.num_classes_per_task)
            if len(self.error_window) >= self.error_window_max_len and current_loss_val > mean + self.error_threshold * std:
                if self.current_task_id_inferred < max_inferred_task_segments -1 : 
                    self.new_task() # This increments self.current_task_id_inferred
                    mean, std = 0, 0 # Reset error stats for the new inferred task
                    print(f"Batch {batch_idx}: New task inferred by loss. Model now on internal inferred task ID: {self.current_task_id_inferred}.")
                else:
                    print(f"Batch {batch_idx}: High loss detected but max task changes ({max_inferred_task_segments -1}) reached. Loss: {current_loss_val:.4f}")
            
            if not (mean == 0 and std == 0 and len(self.error_window) == 0) or current_loss_val !=0: # Avoid division by zero if window empty
                 mean, std = self.new_error(current_loss_val)
            # --- End Anomaly Detection ---
            
            loss_distillation = torch.tensor(0.0, device=device)
            if len(self.old_classifier_heads) > 0 and len(old_outputs_from_model) > 0:
                for i in range(len(old_outputs_from_model)):
                    # output and old_outputs_from_model[i] are both total_classes-dimensional
                    loss_distillation += self.old_loss_weight * self.loss_old(
                        output / self.temperature, 
                        old_outputs_from_model[i].detach() / self.temperature
                    )

            loss: torch.Tensor = loss_new + loss_distillation
            loss.backward()
            self.optimizer.step()
            loading_bar.set_postfix({
                "loss": loss.item(), 
                "loss_new": loss_new.item(), 
                "loss_old": loss_distillation.item(), 
                "mean": f"{mean:.2f}\"",
                "std": f"{std:.2f}\"",
                "inferredTask": self.current_task_id_inferred,
                # "dataTaskID_eg": task_id_tensor[0].item() if task_id_tensor.numel() > 0 else -1 
                }) 
        
        if test_dataset is not None:
            self.eval()
            correct = 0
            total = 0
            # Evaluate using the current total_classes-dimensional head against global labels
            confusion_matrix = torch.zeros(self.total_classes, self.total_classes, dtype=torch.int64, device=device)
            print(f"\\\\nEvaluating with {self.total_classes}-class head.")

            with torch.no_grad():
                loading_bar = tqdm(test_loader, desc=f"Testing ({self.total_classes}-class)", total=len(test_loader))
                for x, y_original_global in loading_bar: 
                    x = x.to(device)
                    y_original_global = y_original_global.to(device) # Global labels (e.g., 0-9)
                    
                    # Logits are from the current total_classes-dimensional head
                    logits, _ = self(x) 
                    
                    total += y_original_global.shape[0]
                    correct += (logits.argmax(dim=1) == y_original_global).sum().item()
                    
                    accuracy = correct / total if total > 0 else 0.0
                    
                    y_flat = y_original_global.view(-1)
                    pred_flat = logits.argmax(dim=1).view(-1)
                    
                    for i in range(y_flat.size(0)):
                        # Ensure indices are within bounds for the confusion matrix
                        true_idx = y_flat[i].item()
                        pred_idx = pred_flat[i].item()
                        if 0 <= true_idx < self.total_classes and 0 <= pred_idx < self.total_classes:
                            confusion_matrix[true_idx, pred_idx] += 1
                        # else: print(f"Warning: CM index out of bounds. True: {true_idx}, Pred: {pred_idx}, Head: {self.total_classes}")
                    
                    loading_bar.set_postfix(accuracy=f"{accuracy:.2%}")
            
            plt.figure(figsize=(10, 10)) # Adjusted for potentially 10 classes
            plt.imshow(confusion_matrix.cpu().numpy(), interpolation='nearest', cmap='Blues')
            plt.colorbar()
            plt.title(f"Confusion Matrix ({self.total_classes} classes)")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.xticks(np.arange(self.total_classes))
            plt.yticks(np.arange(self.total_classes))
            plt.show()

def main():
    cifar_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) 
    ])

    # Configuration for data splitting and model
    total_dataset_classes = 10 # For CIFAR-10
    num_classes_per_task_split = 2 # How SplitCIFAR10 defines tasks, e.g., 5 tasks of 2 classes

    train_dataset = SplitCIFAR10(
        task_duration=10000, # Or however long each task segment should be
        transform=cifar_transform, 
        classes_per_split=num_classes_per_task_split # Tells SplitCIFAR10 how to make tasks
    )
    
    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=cifar_transform)

    # LWFClassifier is initialized with total_classes for its head,
    # and num_classes_per_task for managing inferred task change logic.
    model = LWFClassifier(
        in_out_shape=(3, 32, 32), 
        total_classes=total_dataset_classes,
        num_classes_per_task=num_classes_per_task_split
    ) 
    model.fit(train_dataset, test_dataset)

if __name__ == "__main__":
    # A placeholder for ClassificationHead if it's not defined globally or imported
    class ClassificationHead(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super(ClassificationHead, self).__init__()
            # This is an example structure. Adjust if your ClassificationHead is different.
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_dim, num_classes)
        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
    
    # Make ClassificationHead globally available for LWFClassifier
    globals()['ClassificationHead'] = ClassificationHead
    main()
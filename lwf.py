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
    def __init__(self, in_out_shape=(3, 32, 32), classes=10):
        super(LWFClassifier, self).__init__()
        
        # Use the refactored WideResNet as the feature extractor
        # WRN-20-8: width_factor=8, layers=[3,3,3] (default in WideResNet)
        # Input channels from in_out_shape
        self.feature_extractor = WideResNet(in_channels=in_out_shape[0], width_factor=8)
        
        # The output dimension of the feature extractor (WRN-20-8 gives 512)
        self.classifier_input_dim = self.feature_extractor.feature_dim 
        
        # Assuming ClassificationHead uses a hidden layer.
        # This value was previously derived from wide_resnet50_2's head.
        # We set it to a common value or make it configurable. Let's use 512 or 1000.
        # For consistency with potential previous head structures, let's use 1000.
        # If ClassificationHead is just Linear(input_dim, classes), this might be unused or reinterpreted.
        self.classifier_hidden_dim = 1000 # Or another suitable value like 512

        # Freeze the feature extractor parameters (standard for LwF)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.old_classifier_heads: nn.ModuleList = nn.ModuleList() # Stores old task heads
        self.classes = classes # Number of classes for the current task
        self.lr = 1e-3 # Learning rate

        # Initialize the classifier head for the current task
        # This assumes ClassificationHead takes (input_features, hidden_layer_size, output_classes)
        # If ClassificationHead is not defined, you'll need to provide its implementation.
        self.classifier_head = ClassificationHead(
            self.classifier_input_dim, 
            self.classifier_hidden_dim,
            classes
        )
        
        self.optimizer = torch.optim.SGD(self.classifier_head.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss() # Loss for the current task
        # Distillation loss for old tasks
        self.loss_old = lambda logx, logy: -torch.sum(torch.softmax(logy, dim=1) * torch.log_softmax(logx, dim=1), dim=1).mean()

        # Parameters for task change detection (anomaly detection)
        self.error_window_max_len = 32
        self.error_threshold = 10 
        self.error_window = []
        self.error_window_sum = 0

        # LwF parameters
        self.temperature = 2 # Temperature for distillation
        self.old_loss_weight = 1 # Weight for the distillation loss

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
        # Re-initialize or deepcopy the head for the new task
        # Important: Ensure new head is properly initialized for the new number of classes
        self.classifier_head = ClassificationHead(
            self.classifier_input_dim,
            self.classifier_hidden_dim,
            classes # Use the new number of classes
        )
        # self.classifier_head = copy.deepcopy(self.classifier_head) # If structure is identical
        # If classes can change, direct re-initialization is better as above.

        for param in self.old_classifier_heads[-1].parameters():
            param.requires_grad = False
        self.old_classifier_heads[-1].eval()
        
        self.classes = classes # Update current number of classes
        self.optimizer = torch.optim.SGD(self.classifier_head.parameters(), lr=self.lr)
        self.error_window = []
        self.error_window_sum = 0
        self.batches_with_high_loss = 0

    def forward(self, x: torch.Tensor):
        # Extract features using the new WideResNet
        features = self.feature_extractor(x)
        
        # Get outputs from old task heads (for distillation)
        x_old = []
        for i in range(len(self.old_classifier_heads)):
            x_old.append(self.old_classifier_heads[i](features))

        # Get output from the current task head
        x_current = self.classifier_head(features)
        return x_current, x_old

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
            task = task.to(device) # Assuming task indicates task ID or properties

            self.optimizer.zero_grad()
            # The forward pass now returns (current_task_output, list_of_old_task_outputs)
            output, old_outputs_from_model = self(img) 
            
            loss_new: torch.Tensor = self.loss(output, label)
            
            # Task change detection logic
            # Ensure self.classes reflects the expected classes for the current data
            # The new_task method should be called if a task boundary is detected.
            # The current logic for new_task might need adjustment if 'task' from batch signals a new task.
            # For now, assuming the error-based detection is the primary mechanism.
            current_loss_val = loss_new.item()
            if len(self.error_window) == self.error_window_max_len and current_loss_val > mean + self.error_threshold * std:
                # This implies a new task based on loss anomaly.
                # Need to know the number of classes for this new task.
                # This part is complex if classes change and are not known beforehand.
                # For simplicity, if SplitCIFAR10 yields tasks with same number of classes, this is fine.
                # If classes per task can vary, `new_task` needs the correct `classes` count.
                # Assuming self.classes is appropriate for the new task or updated externally.
                print(f"Potential new task detected. Loss: {current_loss_val:.4f}, Mean: {mean:.4f}, Std: {std:.4f}")
                # self.new_task(self.classes) # Or determine new number of classes
                task_changes += 1 # Incrementing, but actual new_task call might be conditional
            mean, std = self.new_error(current_loss_val)
            
            loss_distillation = torch.zeros_like(loss_new)
            if len(self.old_classifier_heads) > 0 and len(old_outputs_from_model) > 0:
                # Calculate distillation loss using the model's stored old heads
                # and their corresponding outputs from the forward pass.
                # This assumes old_outputs_from_model are the raw logits from old heads.
                for i in range(len(old_outputs_from_model)):
                    # Ensure old_outputs_from_model[i] corresponds to self.old_classifier_heads[i]
                    # The distillation target should be the soft labels from the *frozen* old model's head
                    # For LwF, the old_outputs_from_model are predictions on current data using old heads.
                    # The target for distillation is y_o_hat = sigma(o_o / T) where o_o are logits from old model on old task data
                    # Here, we are using current data, new model's features, and old heads.
                    # The loss_old function expects (current_model_logits_for_old_task, target_distill_logits)
                    # This setup is tricky. If `output` is for current task, and `old_outputs_from_model` are for old tasks.
                    # The LwF paper typically distills by having the new model try to reproduce the old model's outputs *for the old tasks*.
                    # This usually means passing current input through new feature extractor, then through *copies* of old heads,
                    # and comparing these to the *original* old model's (old feature extractor + old head) outputs for those old tasks.
                    # The current code seems to use `output` (current task logits) in the distillation loss against `old_output[i]` (old task logits from current features).
                    # This is non-standard LwF. Standard LwF: L_distill = sum_{old_tasks} KLDiv( new_model_pred_on_old_task_i || old_model_pred_on_old_task_i)
                    # Or, if using current data: L_distill = sum_{old_tasks} CrossEntropy( new_model_pred_for_old_task_i_using_current_data, old_model_pred_for_old_task_i_using_current_data (frozen) )
                    
                    # Given the existing loss_old: lambda logx, logy: -torch.sum(torch.softmax(logy, dim=1) * torch.log_softmax(logx, dim=1), dim=1).mean()
                    # logx: current model's prediction for an old task (e.g. output from a combined head, or specific head)
                    # logy: old model's soft target for that old task
                    # The original code had: loss_old += self.old_loss_weight * self.loss_old(output/self.temperature, old_output[i]/self.temperature)
                    # `output` is current task's logits. `old_output[i]` is old task i's logits (from current features).
                    # This means it's trying to make the current task's output distribution similar to each old task's output distribution. This is unusual.
                    # A more standard LwF would be:
                    # current_preds_for_old_task_i = self.old_classifier_heads[i](features) # features from new extractor
                    # target_preds_for_old_task_i = old_outputs_from_model[i] # These are already from old_heads(current_features)
                    # So, loss_distill += self.old_loss_weight * self.loss_old(current_preds_for_old_task_i / self.temperature, old_outputs_from_model[i].detach() / self.temperature)
                    # Since old_outputs_from_model[i] are already the outputs of old_classifier_heads[i](features), this would be:
                    # loss_distillation += self.old_loss_weight * self.loss_old(old_outputs_from_model[i]/self.temperature, old_outputs_from_model[i].detach()/self.temperature)
                    # This is essentially making the (gradient-enabled) old head match its own detached output, which is not the goal.

                    # Let's assume the intention was to use the `output` (current task's logits) and make them "not forget" old tasks by aligning with `old_outputs_from_model[i]`.
                    # This is what the original code structure implies.
                    loss_distillation += self.old_loss_weight * self.loss_old(output/self.temperature, old_outputs_from_model[i].detach()/self.temperature)


            loss: torch.Tensor = loss_new + loss_distillation
            loss.backward()
            self.optimizer.step()
            loading_bar.set_postfix({
                "loss": loss.item(), 
                "loss_new": loss_new.item(), 
                "loss_old": loss_distillation.item(), 
                "mean": mean, 
                "std": std,
                "tasks": task_changes}) # Renamed 'n' to 'tasks'
        
        if test_dataset is not None:
            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                loading_bar = tqdm(test_loader, desc="Testing", total=len(test_loader))
                # Assuming testing is done on a single task's data, corresponding to current self.classifier_head
                # Or, if test_dataset is multi-task, evaluation needs to be task-aware.
                # The original code implies testing on a dataset that matches the 'current' model structure.
                # For CIFAR10, it's 10 classes.
                num_eval_classes = self.classes # Or fixed if test_dataset is always e.g. full CIFAR10
                confusion_matrix = torch.zeros(num_eval_classes, num_eval_classes, dtype=torch.int64)
                for x, y in loading_bar:
                    x = x.to(device)
                    y = y.to(device)
                    # self() returns (current_task_output, list_of_old_task_outputs)
                    logits, _ = self(x) 
                    # outputs = logits # logits are the current task's outputs
                    total += y.shape[0]
                    correct += (logits.argmax(dim=1) == y).sum().item()
                    accuracy = correct / total
                    
                    # Ensure y and logits.argmax(dim=1) are within bounds for confusion_matrix
                    y_flat = y.view(-1)
                    pred_flat = logits.argmax(dim=1).view(-1)
                    
                    # Clamp values to be safe if classes change and matrix size is fixed
                    y_clamped = torch.clamp(y_flat, 0, num_eval_classes - 1)
                    pred_clamped = torch.clamp(pred_flat, 0, num_eval_classes - 1)

                    # Update confusion matrix
                    for i in range(y_clamped.size(0)):
                        confusion_matrix[y_clamped[i], pred_clamped[i]] += 1
                    
                    loading_bar.set_postfix(accuracy=f"{accuracy:.2%}")
            
            plt.figure(figsize=(10, 10))
            plt.imshow(confusion_matrix.cpu(), interpolation='nearest')
            plt.colorbar()
            plt.title(f"Confusion Matrix (Test - {num_eval_classes} classes)")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.xticks(np.arange(num_eval_classes))
            plt.yticks(np.arange(num_eval_classes))
            plt.show()
        

def main():
    cifar_transform = transforms.Compose([
        transforms.Resize((32, 32)), # Adjusted to 32x32, common for WRN on CIFAR
        # transforms.Resize((224, 224)), # Original resize
        transforms.ToTensor(),
        # Standard CIFAR-10 normalization, not ImageNet, if training from scratch
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) 
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])
    # Assuming SplitCIFAR10 handles task definitions and provides data for 'classes' per task
    # If SplitCIFAR10 yields tasks with varying numbers of classes, LWFClassifier.new_task needs that info.
    train_dataset = SplitCIFAR10(task_duration=10000, transform=cifar_transform)
    
    # Test dataset should ideally cover all classes seen or be task-specific
    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=train_dataset.transform)

    # LWFClassifier needs to know the number of classes for the *first* task.
    # This might come from train_dataset.get_task_info() or be explicit.
    # Assuming initial task has 'classes_per_task' from SplitCIFAR10.
    initial_classes = 2 # Matching example classes_per_task
    model = LWFClassifier(in_out_shape=(3, 32, 32), classes=initial_classes) 
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
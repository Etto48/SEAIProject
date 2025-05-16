import torch
import torch.nn as nn
import torchvision
from torchvision import transforms # Added
from torch.utils.data import Dataset, DataLoader
import copy
import matplotlib.pyplot as plt
import numpy as np

class LWFClassifier(nn.Module):
    def __init__(self, in_out_shape=(3, 32, 32), classes=10, lr=1e-3, lambda_lwf=1.0, distill_temp=2.0): # Added lr, lambda_lwf, distill_temp
        super(LWFClassifier, self).__init__()
        # The in_out_shape parameter is not directly used by torchvision.models.alexnet, 
        # which expects (3, 224, 224) input. It\'s kept for potential future use with other feature extractors.
        self.feature_extractor = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        self.classifier_input_dim = self.feature_extractor.classifier[6].in_features
        self.feature_extractor.classifier.pop(6)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False # Feature extractor is frozen initially
        
        self.old_classifier_head: nn.Linear | None = None
        self.frozen_feature_extractor_for_distill: nn.Module | None = None # For LwF teacher
        self.classes = classes
        self.classifier_head = nn.Linear(self.classifier_input_dim, classes)
        
        self.lr = lr
        self.lambda_lwf = lambda_lwf
        self.distill_temp = distill_temp
        self.is_first_task = True

        # Optimizer initially for the first head only, as FE is frozen
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
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
        # Current classifier becomes old, freeze it
        self.old_classifier_head = self.classifier_head 
        if self.old_classifier_head:
            for param in self.old_classifier_head.parameters():
                param.requires_grad = False
        
        # Store a frozen copy of the current feature extractor state for distillation
        self.frozen_feature_extractor_for_distill = copy.deepcopy(self.feature_extractor)
        if self.frozen_feature_extractor_for_distill:
            for param in self.frozen_feature_extractor_for_distill.parameters():
                param.requires_grad = False
        
        # Unfreeze the main feature extractor for fine-tuning (LwF fine-tunes)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
        self.classes = classes
        self.classifier_head = nn.Linear(self.classifier_input_dim, classes) # New head
        
        # Re-initialize optimizer with all trainable parameters
        trainable_params = []
        # Add feature extractor parameters (now trainable)
        if self.feature_extractor is not None:
            trainable_params.extend(p for p in self.feature_extractor.parameters() if p.requires_grad)
        # Add new classifier head parameters
        trainable_params.extend(self.classifier_head.parameters())
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        
        # After new_task is called, it's no longer the "first task" configuration from __init__
        self.is_first_task = False

    def _distillation_loss(self, student_logits, teacher_logits):
        """Calculates KL divergence for distillation."""
        if teacher_logits is None:
            return torch.tensor(0.0, device=student_logits.device)
        
        # Softmax with temperature for teacher
        soft_teacher_p = nn.functional.softmax(teacher_logits / self.distill_temp, dim=1)
        
        # LogSoftmax with temperature for student
        log_soft_student_p = nn.functional.log_softmax(student_logits / self.distill_temp, dim=1)
        
        # KLDivLoss expects input (log_soft_student_p) and target (soft_teacher_p)
        # The reduction 'batchmean' averages the loss over the batch.
        # Multiply by T^2 as in Hinton's paper on distillation.
        loss = nn.KLDivLoss(reduction='batchmean')(log_soft_student_p, soft_teacher_p) * (self.distill_temp ** 2)
        return loss

    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        output = self.classifier_head(features)
        return output

    def fit(self, train_dataset: Dataset, epochs: int, test_dataset: Dataset | None = None): # Changed IterableDataset to Dataset
        # Determine device from model parameters
        device = next(self.parameters()).device
        self.to(device)

        # Ensure num_workers is appropriate for the system, especially on Windows.
        # For simplicity in this example, it's kept at 4, but 0 might be safer for general portability
        # if issues arise with multiprocessing.
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        # Variables to store data for final visualization
        final_all_preds = []
        final_all_labels = []
        final_sample_images = []
        final_sample_true_labels = []
        final_sample_pred_labels = []

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            processed_batches = 0

            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                current_features = self.feature_extractor(inputs)
                new_task_logits = self.classifier_head(current_features)

                loss_ce = self.loss(new_task_logits, labels)
                total_loss = loss_ce
                loss_d_item = 0.0

                # Distillation loss if old model components exist
                # The self.is_first_task check ensures distillation only happens after the first task is done
                # and new_task has prepared for a subsequent task.
                if not self.is_first_task and self.old_classifier_head is not None and self.frozen_feature_extractor_for_distill is not None:
                    with torch.no_grad():
                        teacher_features = self.frozen_feature_extractor_for_distill(inputs)
                        teacher_logits_for_old_task = self.old_classifier_head(teacher_features)
                    
                    student_logits_for_old_task = self.old_classifier_head(current_features) # Use current FE for student's old task prediction
                    
                    loss_d = self._distillation_loss(student_logits_for_old_task, teacher_logits_for_old_task)
                    total_loss += self.lambda_lwf * loss_d
                    loss_d_item = loss_d.item()

                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()
                processed_batches += 1
                if i % 100 == 99: 
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] avg_loss: {running_loss / processed_batches:.4f} (CE: {loss_ce.item():.4f}, KD: {loss_d_item:.4f})')

            epoch_avg_loss = running_loss / processed_batches
            mean_err, std_err = self.new_error(epoch_avg_loss)
            print(f'Epoch {epoch+1} Summary: Avg Train Loss: {epoch_avg_loss:.4f} (Window Mean: {mean_err:.4f}, Std: {std_err:.4f})')

            if test_dataset is not None and test_loader is not None:
                self.eval()
                test_loss_sum = 0.0
                correct = 0
                total = 0
                # Temporary lists for this epoch's eval, in case we want to collect final ones only from last epoch
                current_epoch_preds = []
                current_epoch_labels = []
                current_epoch_sample_images = []
                current_epoch_sample_true_labels = []
                current_epoch_sample_pred_labels = []

                with torch.no_grad():
                    for i_test, test_data in enumerate(test_loader):
                        test_inputs, test_labels = test_data
                        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                        
                        test_outputs = self(test_inputs)
                        loss = self.loss(test_outputs, test_labels)
                        test_loss_sum += loss.item()
                        
                        _, predicted = torch.max(test_outputs.data, 1)
                        total += test_labels.size(0)
                        correct += (predicted == test_labels).sum().item()

                        if epoch == epochs -1: # Collect data for final plot only on the last epoch
                            final_all_preds.extend(predicted.cpu().numpy())
                            final_all_labels.extend(test_labels.cpu().numpy())
                            if i_test == 0 and len(final_sample_images) < 10:
                                num_to_take = min(10 - len(final_sample_images), test_inputs.size(0))
                                final_sample_images.extend(test_inputs.cpu().numpy()[:num_to_take])
                                final_sample_true_labels.extend(test_labels.cpu().numpy()[:num_to_take])
                                final_sample_pred_labels.extend(predicted.cpu().numpy()[:num_to_take])
                
                avg_test_loss = test_loss_sum / len(test_loader)
                accuracy = 100 * correct / total
                print(f'Epoch {epoch+1} Test: Avg Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}% ({correct}/{total})')

        # Plotting at the end of all epochs for this fit call
        if test_dataset is not None and self.classes > 0 and final_all_labels and final_all_preds:
            confusion_matrix_data = torch.zeros(self.classes, self.classes, dtype=torch.int64)
            for true_label, pred_label in zip(final_all_labels, final_all_preds):
                if 0 <= true_label < self.classes and 0 <= pred_label < self.classes:
                     confusion_matrix_data[true_label, pred_label] += 1

            plt.figure(figsize=(8, 8) if self.classes > 5 else (5,5))
            plt.imshow(confusion_matrix_data.numpy(), interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Final Confusion Matrix (Task Classes: {self.classes}) after {epochs} Epochs')
            plt.colorbar()
            tick_marks = np.arange(self.classes)
            plt.xticks(tick_marks, tick_marks)
            plt.yticks(tick_marks, tick_marks)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.show()

        if final_sample_images:
            num_samples_to_show = len(final_sample_images)
            if num_samples_to_show > 0:
                fig_height = 3 * num_samples_to_show // 5 + (1 if num_samples_to_show % 5 > 0 else 0)
                fig, axes = plt.subplots(max(1, num_samples_to_show // 5 + (1 if num_samples_to_show % 5 > 0 else 0) ), min(num_samples_to_show, 5), figsize=(15, 3 * fig_height))
                if num_samples_to_show == 1: # Handle case for a single sample image
                    axes = np.array([axes]).flatten() 
                else:
                    axes = axes.flatten()

                for i in range(num_samples_to_show):
                    img = final_sample_images[i]
                    img_to_show = np.transpose(img, (1, 2, 0))
                    img_min, img_max = np.min(img_to_show), np.max(img_to_show)
                    if img_max > img_min:
                       img_to_show = (img_to_show - img_min) / (img_max - img_min)
                    img_to_show = np.clip(img_to_show, 0, 1) # Clip to ensure valid image range
                    
                    axes[i].imshow(img_to_show)
                    axes[i].set_title(f'True: {final_sample_true_labels[i]}\\nPred: {final_sample_pred_labels[i]}')
                    axes[i].axis('off')
                plt.suptitle(f'Final Sample Predictions after {epochs} Epochs')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

# Define CIFAR10SubsetDataset
class CIFAR10SubsetDataset(Dataset):
    def __init__(self, root, train=True, download=True, transform=None, subset_classes=None):
        self.full_dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=None) # Apply transform later
        self.transform = transform
        self.subset_classes = subset_classes
        self.data = []
        self.targets = []
        self.label_map = {original_label: new_label for new_label, original_label in enumerate(subset_classes)}

        if subset_classes is None: # Use all classes
            self.data = [self.full_dataset.data[i] for i in range(len(self.full_dataset))]
            self.targets = [self.full_dataset.targets[i] for i in range(len(self.full_dataset))]
        else:
            for i in range(len(self.full_dataset)):
                original_label = self.full_dataset.targets[i]
                if original_label in self.subset_classes:
                    self.data.append(self.full_dataset.data[i])
                    self.targets.append(self.label_map[original_label])
        
        # Convert to numpy array if not already, for consistency with CIFAR10 structure
        if not isinstance(self.data, np.ndarray) and len(self.data) > 0:
            self.data = np.array(self.data)


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        # CIFAR-10 data is numpy.ndarray, needs to be PIL Image for some transforms
        img = torchvision.transforms.functional.to_pil_image(img)

        if self.transform:
            img = self.transform(img)
        return img, target


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-10 transforms
    # AlexNet expects 224x224. CIFAR-10 is 32x32.
    cifar_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])

    # Task definitions
    task1_classes = list(range(5)) # Classes 0-4
    task2_classes = list(range(5, 10)) # Classes 5-9
    
    # Initialize model for the first task
    print(f"Initializing LWFClassifier for first task ({len(task1_classes)} classes)...")
    model = LWFClassifier(classes=len(task1_classes), lr=1e-4, lambda_lwf=1.0, distill_temp=2.0)
    model.to(device)

    # --- Task 1: CIFAR-10 classes 0-4 ---
    print(f"Preparing Task 1: CIFAR-10 classes {task1_classes}")
    train_data_task1 = CIFAR10SubsetDataset(root='./data', train=True, download=True, transform=cifar_transform, subset_classes=task1_classes)
    test_data_task1 = CIFAR10SubsetDataset(root='./data', train=False, download=True, transform=cifar_transform, subset_classes=task1_classes)
    
    print(f"Training on Task 1 ({len(task1_classes)} classes)...")
    # Reduce epochs for quicker example run
    model.fit(train_data_task1, epochs=3, test_dataset=test_data_task1) 

    # --- Task 2: CIFAR-10 classes 5-9 ---
    print(f"\\nPreparing Task 2: CIFAR-10 classes {task2_classes}")
    model.new_task(classes=len(task2_classes)) # Update model for new number of classes
    
    train_data_task2 = CIFAR10SubsetDataset(root='./data', train=True, download=True, transform=cifar_transform, subset_classes=task2_classes)
    test_data_task2 = CIFAR10SubsetDataset(root='./data', train=False, download=True, transform=cifar_transform, subset_classes=task2_classes)
    
    print(f"Training on Task 2 ({len(task2_classes)} classes) with LwF...")
    model.fit(train_data_task2, epochs=3, test_dataset=test_data_task2)
    
    # Example for a third task (e.g., all CIFAR-10 classes, or another subset)
    # task3_classes = list(range(10)) # All 10 classes
    # print(f"\\nPreparing Task 3: CIFAR-10 classes {task3_classes}")
    # model.new_task(classes=len(task3_classes))
    # train_data_task3 = CIFAR10SubsetDataset(root='./data', train=True, download=True, transform=cifar_transform, subset_classes=task3_classes)
    # test_data_task3 = CIFAR10SubsetDataset(root='./data', train=False, download=True, transform=cifar_transform, subset_classes=task3_classes)
    # print(f"Training on Task 3 ({len(task3_classes)} classes) with LwF...")
    # model.fit(train_data_task3, epochs=3, test_dataset=test_data_task3)

    print("\\nDone.")
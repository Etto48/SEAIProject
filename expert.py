import torch
import torch.random as random
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class ExpertMLP(nn.Module):
	def __init__(self, input_feature: int, hidden_features: int, output_features: int):
		super(ExpertMLP, self).__init__()
		
		self.model = nn.Sequential(
			nn.Conv2d(1, input_feature, kernel_size=3, padding=1, padding_mode='reflect'),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(input_feature, input_feature, kernel_size=3, padding=1, padding_mode='reflect'),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d((2, 2)),
			nn.Flatten(),
			nn.Linear(2*2*input_feature, hidden_features),
			nn.ReLU(),
			nn.Linear(hidden_features, hidden_features),
			nn.ReLU(),
			nn.Linear(hidden_features, output_features)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)
	
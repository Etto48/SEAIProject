import torch
import torch.random as random
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm

class ExpertMLP(nn.Module):
	def __init__(self, input_feature: int, hidden_features: int, output_features: int):
		super(ExpertMLP, self).__init__()
		
		self.model = nn.Sequential(
			nn.Linear(input_feature, hidden_features),
			nn.ReLU(),
			nn.Linear(hidden_features, hidden_features),
			nn.ReLU(),
			nn.Linear(hidden_features, output_features)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)
	
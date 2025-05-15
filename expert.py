import torch
import torch.random as random
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class ExpertMLP(nn.Module):
	def __init__(self, input_feature: int, depth: int, hidden_features: int, output_features: int):
		super(ExpertMLP, self).__init__()
		
		self.model = nn.Sequential()
		for i in range(depth):
			in_features = input_feature if i == 0 else hidden_features
			out_features = hidden_features
			nn.Linear(in_features, out_features),
			nn.BatchNorm1d(out_features),
			nn.ReLU(),
		nn.Linear(hidden_features, output_features)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)
	
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class GateAutoencoder(nn.Module):
    def __init__(self, in_out_shape=(3,32,32),  depth=3, ff_depth=3, hidden_dim=128, latent_dim=128):
        super(GateAutoencoder, self).__init__()
        self.input_features = in_out_shape[0]
        self.depth = depth
        self.ff_depth = ff_depth
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        latent_width, latent_height = in_out_shape[1] // (2**depth), in_out_shape[2] // (2**depth)
        self.encoder.append(nn.Conv2d(self.input_features, hidden_dim, kernel_size=3, padding=1, padding_mode='reflect'))
        for i in range(depth):
            self.encoder.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, padding=1, stride=2, padding_mode='reflect'))
        self.encoder.append(nn.Flatten())
        for i in range(ff_depth):
            if i == 0:
                in_features = hidden_dim * latent_width * latent_height
            else:
                in_features = hidden_dim
            if i == ff_depth - 1:
                out_features = latent_dim
            else:
                out_features = hidden_dim
            self.encoder.append(nn.Linear(in_features, out_features))
            self.encoder.append(nn.BatchNorm1d(out_features))
            if i != ff_depth - 1:
                self.encoder.append(nn.ReLU())
            else:
                self.encoder.append(nn.Tanh())

        for i in range(ff_depth):
            if i == 0:
                in_features = latent_dim
            else:
                in_features = hidden_dim
            if i == ff_depth - 1:
                out_features = hidden_dim * latent_width * latent_height
            else:
                out_features = hidden_dim
            self.decoder.append(nn.Linear(in_features, out_features))
            self.decoder.append(nn.BatchNorm1d(out_features))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Unflatten(1, (hidden_dim, latent_width, latent_height)))

        for i in range(depth):
            self.decoder.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, padding=1, stride=2))
        self.decoder.append(nn.Conv2d(hidden_dim, self.input_features, kernel_size=3, padding=1, padding_mode='reflect'))
        self.decoder.append(nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent
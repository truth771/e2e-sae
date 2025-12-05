import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE_Local(nn.Module):
    """Sparse Autoencoder to reconstruct activations of a particular layer."""
    def __init__(self, d_model, d_dict, sparsity_param=0.05):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_dict)
        self.decoder = nn.Linear(d_dict, d_model)

        self.phi = sparsity_param / d_dict

        # might need dictionary normalization here?

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z
    
    def loss(self, x, z, x_reconstructed):
        return F.mse_loss(x_reconstructed, x) + self.phi * torch.mean(z)
    
    def L0_norm(self, z, threshold=1e-4):
        return torch.sum(z > threshold, dim=-1).float().mean()
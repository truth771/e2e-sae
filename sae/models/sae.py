from typing import Optional, Literal
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import losses

@dataclass
class SAEParams:
    sae_layer: Optional[int] = None
    sae_dict_size: Optional[int] = None
    sae_type: Optional[Literal["local", "e2e", "e2e + ds"]] = None

def init_sae_params(self, sae_params, dim):
    self.sae_type = sae_params.sae_type
    if sae_params.sae_type is not None:
        assert sae_params.sae_layer is not None, "sae_layer must be given a value if sae_layer has one"
        assert sae_params.sae_dict_size is not None, "sae_dict_size must be given a value if sae_layer has one"
        self.sae_layer = sae_params.sae_layer
        self.sae = SAE_Local(dim, sae_params.sae_dict_size)
    else:
        self.sae = lambda x: x


class SAE_Local(nn.Module):
    """Sparse Autoencoder to reconstruct activations of a particular layer."""
    def __init__(self, d_model, d_dict, sparsity_param=0.05):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_dict)
        self.decoder = nn.Linear(d_dict, d_model)

        self.phi = sparsity_param / d_dict

    # normalize every time after update
    def normalize_dictionary(self):
        with torch.no_grad():
            W = self.encoder.weight.data  # shape: (d_dict, d_model)
            W_norm = W.norm(p=2, dim=1, keepdim=True)  # shape: (d_dict, 1)
            W = W / W_norm
            self.encoder.weight.data = W

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z, losses.l1_penalty(z, self.phi * x.size(-1), x.size(-1))
    
    def loss(self, x, z, x_reconstructed):
        return F.mse_loss(x_reconstructed, x) + self.phi * torch.mean(z)
    
    def L0_norm(self, z, threshold=1e-4):
        return torch.sum(z > threshold, dim=-1).float().mean()

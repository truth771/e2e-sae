from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader

def eval_downstream_recon(model: nn.Module, dataloader: DataLoader, device: torch.device, sae_type: str = "e2e + ds") -> Dict[str, float]:
    # evaluate downstream reconstruction for a sae modes
    # ||a^(k) - ahat^(k)||^2_2 over all downstream leayer k > lambda
    # for local just mse, for e2e+ds, only over downstream layers

    old_sae_type = getattr(model.transformer, "sae_type", None)
    model.transformer.sae_type = sae_type

    model.eval()
    total_mse = 0.0
    total_sparsity = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids: torch.Tensor = batch["input_ids"].to(device)
            
            logits, presents, (mse_losses, sparsity_penalties) = model(input_ids=input_ids)
            num_tokens = input_ids.numel()
            total_mse += mse_losses.sum().item() * num_tokens
            total_sparsity += sparsity_penalties.sum().item() * num_tokens
            total_tokens += num_tokens
    
    model.transformer.sae_type = old_sae_type
    if total_tokens == 0:
        return {"downstream_recon_mse": 0.0, "downstream_recon_sparsity": 0.0}
    
    return {
        "downstream_recon_mse": total_mse / total_tokens,
        "downstream_recon_sparsity": total_sparsity / total_tokens,
    }





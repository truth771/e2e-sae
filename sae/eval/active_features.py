from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


tolerance = 0.0001
def eval_l0_and_active_features(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate the average number of active features (L0 norm) in SAE."""

    model.eval()
    total_active = 0.0
    total_datapoints = 0

    batch = next(iter(dataloader))
    _, _, (_, activations, _) = model(input_ids=batch["input_ids"].to(device))
    active_features = torch.zeros(activations.shape[-1]).to(device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)

            logits, presents, (mse_losses, activations, sparsity_penalties) = model(input_ids=input_ids)

            active_in_batch = (activations.abs() > tolerance)
            active_features += active_in_batch.sum(dim=(0, 1))
            total_active += active_in_batch.sum().item()
            total_datapoints += activations.size(0) * activations.size(1)  # batch size * context length

    if total_datapoints == 0:
        return 0.0, 0.0
    return total_active / total_datapoints, (active_features > 0).sum().item()

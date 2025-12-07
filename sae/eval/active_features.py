import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def eval_l0(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate the average number of active features (L0 norm) in SAE."""

    model.eval()
    total_active = 0.0
    total_datapoints = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)

            logits, presents, (mse_losses, activations, sparsity_penalties) = model(input_ids=input_ids)
            activations_flat = activations.view(activations.size(0), -1)
            active_counts = (activations_flat != 0).sum(dim=1)
            total_active += active_counts.sum().item()
            total_datapoints += activations.size(0)

    if total_datapoints == 0:
        return 0.0
    return total_active / total_datapoints

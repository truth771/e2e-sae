from typing import Literal

import torch
import torch.nn.functional as F

from ..models import get_model, SAEParams
from .datasets import get_openwebtext_dataloaders, OpenWebTextConfig


def train(model_str: Literal["gpt2", "llama"], sae_params: SAEParams):
    train_loader, _ = get_openwebtext_dataloaders(OpenWebTextConfig(), 1)
    normal_model, _ = get_model(model_str, SAEParams())
    model, trainable_params = get_model(model_str, sae_params)
    
    sparsity_weight = 4.0
    mse_weight = 2.5

    optimizer = torch.optim.Adam(trainable_params, lr=0.005)

    for batch in train_loader:
        optimizer.zero_grad()
        logits, *_, (mse_losses, _, sparsity_param) = model(batch["input_ids"])
        normal_logits, *_ = normal_model(batch["input_ids"])

        kl_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.log_softmax(normal_logits, dim=-1),
            log_target=True,
            reduction="batchmean",
        )

        match sae_params.sae_type:
            case "e2e":
                total_loss = kl_loss + sparsity_weight * sparsity_param
            case "e2e + ds":
                total_loss = kl_loss + sparsity_weight * sparsity_param + mse_weight * mse_losses
            case "local":
                total_loss = mse_losses
            case _:
                raise ValueError("Unexpected sae_type encountered.")

        total_loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train("gpt2", SAEParams(6, 1000, "local"))


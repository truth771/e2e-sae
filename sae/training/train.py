from typing import Literal, Container

import tqdm 
import torch
import torch.nn.functional as F

from ..models import get_model, SAEParams
from .datasets import get_openwebtext_dataloaders, OpenWebTextConfig


def train(model_str: Literal["gpt2", "llama"], sae_params: SAEParams, 
          batch_size: int = 32, n_epochs: int = 20, checkpoint_epochs: Container = {},
          sparsity_weight: float = 4.0, mse_weight: float = 2.5):

    train_loader, _ = get_openwebtext_dataloaders(OpenWebTextConfig(), batch_size)
    normal_model, _ = get_model(model_str, SAEParams())
    model, trainable_params = get_model(model_str, sae_params)

    print(model.device())
    
    sparsity_weight = 4.0
    mse_weight = 2.5

    optimizer = torch.optim.Adam(trainable_params, lr=0.005)

    for epoch in range(n_epochs):
        for batch in tqdm.tqdm(train_loader, desc=f"epoch {epoch + 1}"):
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
        if epoch in checkpoint_epochs:
            torch.save(model.state_dict(), f"model-{sae_params.sae_layer}-{sae_params.sae_dict_size}-{sae_params.sae_type}-epoch-{epoch}.pth")

if __name__ == "__main__":
    train("gpt2", SAEParams(6, 1000, "local"))


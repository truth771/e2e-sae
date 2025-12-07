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

    model.train()

    try:
        model.to("cuda")
        normal_model.to("cuda")
    except:
        print("WARNING: running model on CPU")
    try:

        device = trainable_params[0].device
        
        sparsity_weight = 4.0
        mse_weight = 2.5

        optimizer = torch.optim.Adam(trainable_params, lr=0.005)

        for epoch in range(n_epochs):
            for batch in tqdm.tqdm(train_loader, desc=f"epoch {epoch + 1}"):
                optimizer.zero_grad()
                logits, *_, (mse_losses, _, sparsity_param) = model(batch["input_ids"].to(device))
                normal_logits, *_ = normal_model(batch["input_ids"].to(device))

                kl_loss = 0
                if sae_params.sae_type != "local":
                    kl_loss = F.kl_div(
                        F.log_softmax(logits, dim=-1),
                        F.log_softmax(normal_logits, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    )
                    assert kl_loss.requires_grad and kl_loss.grad_fn is not None

                assert sparsity_param.requires_grad and sparsity_param.grad_fn is not None

                if sae_params.sae_type != "e2e":
                    assert mse_losses.requires_grad and mse_losses.grad_fn is not None

                match sae_params.sae_type:
                    case "e2e":
                        total_loss = kl_loss + sparsity_weight * sparsity_param
                    case "e2e + ds":
                        total_loss = kl_loss + sparsity_weight * sparsity_param + mse_weight * mse_losses
                    case "local":
                        total_loss = sparsity_weight * sparsity_param + mse_weight * mse_losses
                    case _:
                        raise ValueError("Unexpected sae_type encountered.")

                total_loss.backward()
                optimizer.step()
                model.transformer.sae.normalize_dictionary()

            if epoch in checkpoint_epochs:
                torch.save(model.state_dict(), f"model-{sae_params.sae_layer}-{sae_params.sae_dict_size}-{sae_params.sae_type}-sp{sparsity_weight}-mse{mse_weight}-epoch-{epoch}.pth")
    finally:
        return model

if __name__ == "__main__":
    train("gpt2", SAEParams(6, 1000, "local"))


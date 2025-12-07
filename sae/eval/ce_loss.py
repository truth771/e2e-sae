import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


def ce_loss_increase(original_preds: torch.Tensor, sae_preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    original_preds_flat = original_preds.view(-1, original_preds.size(-1))
    sae_preds_flat = sae_preds.view(-1, sae_preds.size(-1))
    targets_flat = targets.view(-1)

    ce_original = nn.CrossEntropyLoss(original_preds_flat, targets_flat)
    ce_sae = nn.CrossEntropyLoss(sae_preds_flat, targets_flat)

    increase = ce_sae - ce_original

    return increase.item(), ce_original.item(), ce_sae.item()


def eval_ce_loss_increase(original_model: nn.Module, sae_model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    original_model.eval()
    sae_model.eval()
    total_ce_increase = 0.0
    total_ce_original = 0.0
    total_ce_sae = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            original_logits, _, (_, _, _) = original_model(input_ids=input_ids)
            sae_logits, _, (_, _, _) = sae_model(input_ids=input_ids)

            ce_increase, ce_original, ce_sae = ce_loss_increase(original_logits, sae_logits, targets)
            num_tokens = targets.numel()
            total_ce_increase += ce_increase * num_tokens
            total_ce_original += ce_original * num_tokens
            total_ce_sae += ce_sae * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return {"ce_loss_increase": 0.0, "ce_original": 0.0, "ce_sae": 0.0}

    return {
        "ce_loss_increase": total_ce_increase / total_tokens,
        "ce_original": total_ce_original / total_tokens,
        "ce_sae": total_ce_sae / total_tokens,
    }
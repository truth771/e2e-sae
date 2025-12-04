from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# helper functions

def  l1_penalty(z: torch.Tensor, lambdas: float, d_hidden: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns: 
        l1_term: scaled l1 penalty: phi * mean(|z|)
        l1_raw: mean(|z|) without scaling 
    """
    phi = lambdas / float(d_hidden)
    l1_raw = z.abs().mean()
    l1_term = phi * l1_raw
    return l1_term, l1_raw

def apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Apply attention mask to per token quantities
    Return: scalar mean over masked tokens
    """

    if mask is None:
        return x.mean()
    
    mask_f = mask.to(x.dtype)
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)

    x_masked = x * mask_f
    denom = mask_f.sum()
    if denom <= 0:
        return x.mean()
    return x_masked.sum() / denom


# Loss Functions

def local_loss(a_l: torch.Tensor, ahat_l: torch.Tensor, z: torch.Tensor, lambdas: float, d_hidden: int, 
               attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    """
    l_local = ||a_l - ahat_l||_2^2 + phi * ||z||_1
    """

    mse_per_token = ((a_l - ahat_l) ** 2).mean(dim=-1)
    recon_mse = apply_mask(mse_per_token, attention_mask)
    l1_term, l1_raw = l1_penalty(z, lambdas, d_hidden)
    loss = recon_mse + l1_term

    metrics = {
        "recon_mse": recon_mse.detach(),
        "l1": l1_raw.detach(),
        "l1_scaled": l1_term.detach(),
        "loss": loss.detach()
    }

    return loss, metrics



def e2e_loss(logits_orig: torch.Tensor, logits_sae: torch.Tensor, z: torch.Tensor, lambdas: float, d_hidden: int, 
             attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    """
    l_e2e = KL(yhat, y) + phi * ||z||_1
    KL(yhat, y) = E_x [KL(phat || p)]
    phat = S(yhat)
    p = S(y)
    """

    log_probs_sae = F.log_softmax(logits_sae, dim=-1)
    log_probs_orig = F.log_softmax(logits_orig, dim=-1)
    probs_sae = log_probs_sae.exp()

    kl_token = (probs_sae * (log_probs_sae - log_probs_orig)).sum(dim=-1)
    kl = apply_mask(kl_token, attention_mask)

    l1_term, l1_raw = l1_penalty(z, lambdas, d_hidden)
    loss = kl + l1_term

    metrics = {
        "kl": kl.detach(),
        "l1": l1_raw.detach(),
        "l1_scaled": l1_term.detach(),
        "loss": loss.detach()
    }

    return loss, metrics


def e2e_downstream_loss(logits_orig: torch.Tensor, logits_sae: torch.Tensor, downstream_orig: Dict[int, torch.Tensor], 
                        downstream_sae: Dict[int, torch.Tensor], z: torch.Tensor, lambdas: float, d_hidden: int, beta: float,
                        attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    """
    l_e2e_ds = KL(yhat, y) + phi * ||z||_1 + beta * (1/(L-1)) * sum_{k=l+1}^{L-1} || ahat^k - a^k ||^2_2
    """

    log_probs_sae = F.log_softmax(logits_sae, dim=-1)
    log_probs_orig = F.log_softmax(logits_orig, dim=-1)
    probs_sae = log_probs_sae.exp()

    kl_token = (probs_sae * (log_probs_sae - log_probs_orig)).sum(dim=-1)
    kl = apply_mask(kl_token, attention_mask)

    l1_term, l1_raw = l1_penalty(z, lambdas, d_hidden)
    mse_terms = []

    for i in downstream_orig.keys():
        a_i = downstream_orig[i]
        ahat_i = downstream_sae[i]
        mse_per_token_i = ((a_i - ahat_i) ** 2).mean(dim=-1)
        mse_k = apply_mask(mse_per_token_i, attention_mask)
        mse_terms.append(mse_k)

    if len(mse_terms) > 0:
        mse_downstream = torch.stack(mse_terms).mean()
    else:
        mse_downstream = torch.tensor(0.0, device=logits_orig.device)


    loss = kl + l1_term + beta * mse_downstream

    metrics = {
        "kl": kl.detach(),
        "l1": l1_raw.detach(),
        "l1_scaled": l1_term.detach(),
        "mse_downstream": mse_downstream.detach(),
        "loss": loss.detach()
    }

    return loss, metrics

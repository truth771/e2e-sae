from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

def normalize(d: torch.Tensor) -> torch.Tensor:
    # l2 normalization of a dictionary d
    i = 1e-8
    norms = d.norms(p=2, dim=1, keepdim=True).clamp_min(i)
    return d / norms


def in_dictionary_cos(d: torch.Tensor) -> Dict[str, torch.Tensor]:
    # for each feature i nearest neighbor cos similarity
    # cos_ij = <d_i, d_j> / (||d_i|| * ||d_j||)
    # nn_cos_i = max{i != j} cos_ij

    # cos similarity matrix
    d_norm = normalize(d)
    C = d_norm @ d_norm.T
    
    #ignore self similarity
    features = C.size(0)
    diag = torch.eye(features, device=C.device, dtype=torch.bool)
    C.masked_fill_(diag, -1e9)

    # nearest neighor
    nn_cos, _ = C.max(dim=1)
    
    mean_nn_cos = nn_cos.mean()
    max_nn_cos = nn_cos.max()

    return {
        "nn_cos" : nn_cos,
        "mean_nn_cos": mean_nn_cos,
        "max_nn_cos": max_nn_cos
    }

# Skipping comparing seeds part

def cross_type_similarity(d_source: torch.Tensor, d_target: torch.Tensor) -> Dict[str, torch.Tensor]:
    # compare SAEs features
    # sim i = max cos(d_source, d_target)

    d_source_norm = normalize(d_source)
    d_target_norm = normalize(d_target)

    # cos similarity matrix
    C = d_source_norm @ d_target_norm.T

    # for each source feature, get best matching target feature
    best_match_cos, best_match_idx = C.max(dim=1)

    mean_best_match_cos = best_match_cos.mean()
    max_best_match_cos = best_match_cos.max()

    return {
        "best_match_cos": best_match_cos,
        "mean_best_match_cos": mean_best_match_cos,
        "max_best_match_cos": max_best_match_cos
    }


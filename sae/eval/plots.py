import matplotlib.pyplot as plt
import numpy as np
import torch
from sae.eval.ce_loss import evaluate_ce_loss_increase
from sae.eval.active_features import eval_l0
from sae.eval.geometry import cross_type_similarity
from sae.models import get_model, SAEParams
from sae.training.train import train

def plot_pareto_curve(model_str, sae_params: SAEParams, val_dataloader, device: torch.device):
    sparsity_params = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1]
    sae_types = ["local", "e2e", "e2e + ds"]

    plt.figure()
    for sae_type in sae_types:
        ce_losses = []
        active_features = []
        sae_params.sae_type = sae_type
        for sp in sparsity_params:
            model = train(model_str=model_str, sae_params=sae_params, sparsity_weight=sp)
            ce_loss = evaluate_ce_loss_increase(model, val_dataloader, device)
            l0_norm = eval_l0(model, val_dataloader, device)
            ce_losses.append(ce_loss)
            active_features.append(l0_norm)
        plt.plot(active_features, ce_losses, marker='o', label=sae_type)

    plt.xlabel('L0')
    plt.ylabel('Cross-Entropy Loss Increase')
    plt.title('Pareto Curves')
    plt.legend()
    plt.show()

def plot_cosine_similarity(model1: str, model2: str, d_source: torch.Tensor, d_target: torch.Tensor):
    similarities = cross_type_similarity(d_source, d_target)

    plt.figure()
    plt.hist(similarities["best_match_cos"].cpu().numpy(), bins=50)
    plt.xlabel('Cosine Similarity')
    plt.ylabel(model1 + ' to ' + model2 + ' Dictionary Similarity')
    plt.title('Cosine Similarity between Source and Target Dictionaries')
    plt.show()

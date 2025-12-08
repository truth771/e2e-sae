import matplotlib.pyplot as plt
import numpy as np
import torch
from sae.eval.ce_loss import eval_ce_loss_increase
from sae.eval.active_features import eval_l0_and_active_features
from sae.eval.geometry import cross_type_similarity
from sae.models import get_model, SAEParams
from sae.training.train import train

def plot_pareto_curves(model_str, sae_params: SAEParams, train_dataloader, val_dataloader, device: torch.device):
    sparsity_params = [10e-4, 10e-2, 10e0, 10e1]
    sae_types = ["local", "e2e", "e2e + ds"]

    normal_model = get_model('gpt2', SAEParams())[0].to(device)

    plt.figure()

    l0_norms, active_features, ce_losses = ({k: []  for k in sae_types} for _ in range(3))
    for sae_type in sae_types:
        sae_params.sae_type = sae_type
        for sp in sparsity_params:
            model = train(model_str=model_str, sae_params=sae_params, train_loader=train_dataloader, sparsity_weight=sp, n_epochs=1)
            ce_losses[sae_type].append(eval_ce_loss_increase(normal_model, model, val_dataloader, device)["ce_loss_increase"])
            l0, act = eval_l0_and_active_features(model, val_dataloader, device)
            active_features[sae_type].append(act)
            l0_norms[sae_type].append(l0)

    for sae_type, norms in l0_norms.items():
        plt.plot(norms, ce_losses[sae_type], marker='o', label=sae_type)
    plt.xlabel('L0')
    plt.ylabel('Cross-Entropy Loss Increase')
    plt.title('Pareto Curves')
    plt.savefig("pareto_l0.png")
    plt.legend()
    plt.show()

    plt.clf()
    for sae_type, acts in active_features.items():
        plt.plot(acts, ce_losses[sae_type], marker='o', label=sae_type)
    plt.xlabel('Total Active Features')
    plt.ylabel('Cross-Entropy Loss Increase')
    plt.title('Pareto Curves')
    plt.savefig("pareto_total_active.png")
    plt.legend()
    plt.show()


def plot_cosine_similarity(model1: str, model2: str, d_source: torch.Tensor, d_target: torch.Tensor):
    similarities = cross_type_similarity(d_source, d_target)

    plt.figure()
    plt.hist(similarities["best_match_cos"].cpu().numpy(), bins=50)
    plt.xlabel('Cosine Similarity')
    plt.ylabel(model1 + ' to ' + model2 + ' Dictionary Similarity')
    plt.title('Cosine Similarity between Source and Target Dictionaries')
    plt.savefig(f"cosine_similarity_{model1}_to_{model2}.png")
    plt.show()

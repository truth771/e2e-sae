from sae.models import get_model
from sae.training.train import train, SAEParams
from sae.training.datasets import get_openwebtext_dataloaders, OpenWebTextConfig
from sae.eval.plots import plot_cosine_similarity

if __name__ == "__main__":
    train_loader, val_loader = get_openwebtext_dataloaders(OpenWebTextConfig(), 4)
    model_local = train("gpt2", SAEParams(6, 60 * 768, "local"), train_loader)
    model_e2e = train("gpt2", SAEParams(6, 60 * 768, "e2e"), train_loader)
    model_e2e_ds = train("gpt2", SAEParams(6, 60 * 768, "e2e + ds"), train_loader)

    d_local = model_local.transformer.sae.encoder.weight.data
    d_e2e = model_e2e.transformer.sae.encoder.weight.data
    d_e2e_ds = model_e2e_ds.transformer.sae.encoder.weight.data

    plot_cosine_similarity("local", "e2e", d_local, d_e2e)
    plot_cosine_similarity("local", "e2e + ds", d_local, d_e2e_ds)
    plot_cosine_similarity("e2e", "e2e + ds", d_e2e, d_e2e_ds)
from typing import Literal

from ..models import get_model, SAEParams
from .datasets import get_openwebtext_dataloaders, OpenWebTextConfig


def train(model_str: Literal["gpt2", "llama"], sae_params: SAEParams):
    train_loader, _ = get_openwebtext_dataloaders(OpenWebTextConfig(), 32)
    model = get_model(model_str, sae_params)

    for batch in train_loader:
        print(batch)

if __name__ == "__main__":
    train("gpt2", SAEParams(6, 1000, "local"))


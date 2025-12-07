from typing import Literal
import torch

from .sae import SAEParams
from .gpt2_sae import GPT2LMHeadModel
from .gpt2_utils import GPT2Config, load_weight

def get_model(model_type: Literal["gpt2", "llama"], sae_params: SAEParams):
    if model_type == "gpt2":
        model = GPT2LMHeadModel(GPT2Config(), sae_params)
        state_dict = torch.load('gpt2-pytorch_model.bin')
        load_weight(model, state_dict)
        unfreeze_parameters = []
        if sae_params.sae_type is not None:
            unfreeze_parameters = [*model.transformer.sae.parameters()]
    elif model_type == "llama":
        raise NotImplementedError()
    else:
        raise ValueError("Expected one of 'gpt2' or 'llama' for model type.")

    for param in model.parameters():
        param.requires_grad = False

    for param in unfreeze_parameters:
        param.requires_grad = True

    return model, unfreeze_parameters

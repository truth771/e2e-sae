from typing import Optional, Literal
import torch

from .llama_sae import Transformer, SAEParams
from .gpt2_sae import GPT2LMHeadModel
from .gpt2_utils import GPT2Config, load_weight

def get_model(model_type: Literal["gpt2", "llama"], sae_params: SAEParams):
    if model_type == "gpt2":
        model = GPT2LMHeadModel(GPT2Config(), sae_params)
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        load_weight(model, state_dict)
    # elif model_type == "llama":
    #     pass
    else:
        raise ValueError("Expected one of 'gpt2' or 'llama' for model type.")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.sae.parameters():
        param.required_grad = True

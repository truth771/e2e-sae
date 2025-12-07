from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer



""""

File for loading OpenWebText dataset for language model training for GPT2


"""


@dataclass
class OpenWebTextConfig:
    # config for OpenWebText datasets

    tokenizer_name: str = "gpt2"
    context_length: int = 1024
    cache_dir: Optional[str] = None

    val_fraction: float = 0.1
    max_train_blocks: Optional[int] = None
    max_val_blocks: Optional[int] = None

    # num_proc: int = 4
    seed: int = 42


class OpenWebTextDataset(Dataset):
    # Flattened, tokenized OpenWebText dataset cut into fixed length blocks
    # each get returns {input_ID: attention_mask}

    def __init__(self, split: str, config: OpenWebTextConfig):
        assert split in {"train", "val"}, "split must be train or val"

        self.config = config
        self.split = split

        # load raw openwebtext dataset 
        # CHANGE LATER TO REAL DATASET

        raw_dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=config.cache_dir)
        raw_dataset = raw_dataset.select(range(5000))

        # split
        num_total = len(raw_dataset)
        num_train = int((1.0 - config.val_fraction) * num_total)
        if split == "train":
            raw_split = raw_dataset.select(range(num_train))
        else:
            raw_split = raw_dataset.select(range(num_train, num_total))

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, cache_dir=config.cache_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer


        # tokenize all text to input ids
        all_ids_list = []
        for example in raw_split["text"]:
            if not example or example.isspace():
                continue
            ids = tokenizer.encode(example, add_special_tokens=False)
            if len(ids) == 0:
                continue
            all_ids_list.extend(ids)

        if len(all_ids_list) == 0:
            raise RuntimeError("No tokens produced")

        all_ids = torch.tensor(all_ids_list, dtype=torch.long)

        # chunk into fixed len blocks
        block_len = config.context_length
        num_blocks = len(all_ids) // block_len

        if split == "train" and config.max_train_blocks is not None:
            num_blocks = min(num_blocks, config.max_train_blocks)
        if split == "val" and config.max_val_blocks is not None:
            num_blocks = min(num_blocks, config.max_val_blocks)

        used_tokens = num_blocks * block_len
        all_ids = all_ids[:used_tokens]

        self.all_ids = all_ids
        self.num_blocks = num_blocks
        self.block_len = block_len


    def __len__(self):
        return self.num_blocks
    
    def __getitem__(self, idx: int):
        start = idx * self.block_len
        end = start + self.block_len
        input_ids = self.all_ids[start:end]

        # full attention over block
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    


def get_openwebtext_dataloaders(config: OpenWebTextConfig, batch_size: int, num_workers: int = 0, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    train_dataset = OpenWebTextDataset(split="train", config=config)
    val_dataset = OpenWebTextDataset(split="val", config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader

# test_openwebtext_dataset.py

import torch
from training.datasets import OpenWebTextConfig, get_openwebtext_dataloaders

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # small test
    dataset_config = OpenWebTextConfig(
        tokenizer_name="gpt2",
        context_length=128,  
        val_fraction=0.001,
        max_train_blocks=100,
        max_val_blocks=10,
        cache_dir=None,
        # num_proc=2,
        seed=0,
    )

    print("Building dataloaders...")
    train_loader, val_loader = get_openwebtext_dataloaders(
        dataset_config,
        batch_size=8,
        num_workers=2,
    )

    print("Loading GPT2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()

    # one batch from train loader
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"] 
    attention_mask = batch["attention_mask"] 

    print(f"Batch input_ids shape: {input_ids.shape}")
    print(f"Batch attention_mask shape: {attention_mask.shape}")

    # decode a single example 
    example_tokens = input_ids[0]
    decoded = tokenizer.decode(example_tokens, skip_special_tokens=True)
    print("\nDecoded example (truncated):")
    print(decoded[:500], "...\n")

    # run gpt to see shapes
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, vocab_size]

    print(f"Logits shape from GPT-2: {logits.shape}")
    print("If you see [batch_size, context_length, vocab_size] above, everything is wired correctly.")


if __name__ == "__main__":
    main()

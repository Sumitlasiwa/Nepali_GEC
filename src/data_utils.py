"""
Dataset loading and preprocessing utilities
"""
from datasets import load_dataset
import torch
import random
import numpy as np

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"✅ Seeds set to {seed}")

def load_and_prepare_dataset(config):
    """
    Load dataset and create train/valid splits
    
    Args:
        config: Config object with dataset settings
    
    Returns:
        DatasetDict with 'train' and 'valid' splits
    """
    print(f"\n📚 Loading dataset: {config.dataset_name}")
    ds = load_dataset(config.dataset_name)
    
    # Select subset if specified
    if config.num_samples:
        ds["train"] = ds["train"].shuffle(seed=config.seed).select(range(config.num_samples))
        print(f"  Using {config.num_samples} samples")
    
    # Create train/valid split
    split_ds = ds["train"].train_test_split(valid_size=config.valid_size, seed=config.seed)
    dataset = {
        "train": split_ds["train"],
        "valid": split_ds["test"]
    }
    
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Valid: {len(dataset['valid'])} samples")
    
    return dataset

def preprocess_dataset(dataset, tokenizer, config):
    """
    Tokenize and prepare dataset for training
    
    Args:
        dataset: Dict with 'train' and 'valid' splits
        tokenizer: Tokenizer instance
        config: Config object
    
    Returns:
        Encoded dataset ready for training
    """
    print("\n⚙️  Preprocessing dataset...")
    
    def preprocess_batch(batch):
        # Add prefix to inputs
        inputs = [config.prefix + inp for inp in batch["incorrect_sentence"]]
        
        # Tokenize inputs
        input_encodings = tokenizer(
            inputs, 
            max_length=config.max_length,
            truncation=True 
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            target_encodings = tokenizer(
                batch["correct_sentence"], 
                max_length=config.max_length,
                truncation=True
            )
        
        # Set labels
        input_encodings["labels"] = target_encodings["input_ids"]
        return input_encodings
    
    # Process both splits
    encoded = {
        "train": dataset["train"].map(preprocess_batch, batched=True),
        "valid": dataset["valid"].map(preprocess_batch, batched=True)
    }
    
    # Set format for PyTorch
    for split in ["train", "valid"]:
        encoded[split].set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    print("  ✅ Preprocessing complete")
    return encoded
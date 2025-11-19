                
import warnings
warnings.filterwarnings("ignore")

import wandb
from math import ceil
from transformers import (
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# Import our custom modules
from config import Config
from metrics import create_compute_metrics
from data_utils import set_seeds, load_and_prepare_dataset, preprocess_dataset
from utils import clear_memory, create_directories, safe_training_check, save_model_safe
from train import setup_model, create_training_args

def main():
    """Main training function"""
    
    # Load config
    config = Config()
    
    print("=" * 60)
    print("🚀 Nepali Grammar Error Correction Training")
    print("=" * 60)
    print(f"Model: {config.model_id}")
    print(f"LoRA: {config.use_lora}")
    print(f"Samples: {config.num_samples or 'Full dataset'}")
    print("=" * 60)
    
    # Setup
    set_seeds(config.seed)
    clear_memory()
    create_directories(config.output_dir)
    
    # Initialize wandb
    wandb.finish()
    wandb.init(
        project=config.wandb_project,
        config=vars(config)
    )
    run_id = wandb.run.id
    
    # Load data
    dataset = load_and_prepare_dataset(config)
    
    # Setup model
    model, tokenizer = setup_model(config)
    
    # Preprocess
    dataset_encoded = preprocess_dataset(dataset, tokenizer, config)
    
    # Create training args
    training_args = create_training_args(config, dataset_encoded, run_id)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True
    )
    
    # Create metrics
    compute_metrics = create_compute_metrics(tokenizer, config)
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        ]
    )
    
        # Safety check
    if not safe_training_check(trainer):
        print("\n❌ Safety checks failed! Fix issues before training.")
        return
    
    # Train!
    print("\n" + "=" * 60)
    print("🏋️  Starting training...")
    print("=" * 60)
    
    try:
        trainer.train()
        print("\n✅ Training complete!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        wandb.finish()
        return
    
    # Save model
    best_model_path = f"{config.output_dir}/best_model"
    save_model_safe(model, tokenizer, best_model_path, use_lora=config.use_lora)
    
    print(f"\n🎉 All done! Model saved to {best_model_path}")
    wandb.finish()

if __name__ == "__main__":
    main()

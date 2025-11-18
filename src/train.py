"""
Main training script for Nepali GEC
Keep this file clean - all logic is in other modules!
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import wandb
from math import ceil
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import our custom modules
from config import Config
from metrics import create_compute_metrics
from data_utils import set_seeds, load_and_prepare_dataset, preprocess_dataset
from utils import clear_memory, create_directories, safe_training_check, save_model_safe

def setup_model(config):
    """Load and configure model based on settings"""
    print(f"\n Lodaing model: {config.model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=False, legacy=False)
    
    if config.use_lora:
        print(" Using LoRA + 8-bit quantization")
        
        # Load with quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_id,
            quantization_config=quantization_config,
            device_map="auto"
        ) 
        
        # Prepare for LoRA
        model = prepare_model_for_kbit_training(model)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Add LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()

    else:
        print("  Using full fine-tuning")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16 if config.use_fp16 else None,
            device_map="auto"
        )
    
    return model, tokenizer

def create_training_args(config, dataset_encoded, run_id):
    """Create training arguments from config"""
    
    # Calculate steps
    steps_per_epoch = ceil(len(dataset_encoded["train"]) / 
                          (config.batch_size * config.gradient_accumulation_steps))
    num_training_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(config.warmup_ratio * num_training_steps)
    
    print(f"\n📊 Training plan:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    return Seq2SeqTrainingArguments(
        output_dir=f"{config.output_dir}/checkpoints",
        num_train_epochs=config.num_epochs,
        
        # Batch & optimization
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        
        # Memory & speed
        fp16=config.use_fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        optim="paged_adamw_8bit" if config.use_lora else "adamw_torch",
        
        # Logging & saving
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        
        # Best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Generation
        predict_with_generate=True,
        generation_max_length=config.max_length,
        generation_num_beams=1,
        
        # Reproducibility
        seed=config.seed,
        data_seed=config.seed,
        
        # Logging
        report_to="wandb",
        run_name=f"{run_id}",
        push_to_hub=False,
        overwrite_output_dir=True,
    )

# def main():
#     """Main training function"""
    
#     # Load config
#     config = Config()
    
#     print("=" * 60)
#     print("🚀 Nepali Grammar Error Correction Training")
#     print("=" * 60)
#     print(f"Model: {config.model_id}")
#     print(f"LoRA: {config.use_lora}")
#     print(f"Samples: {config.num_samples or 'Full dataset'}")
#     print("=" * 60)
    
#     # Setup
#     set_seeds(config.seed)
#     clear_memory()
#     create_directories(config.output_dir)
    
#     # Initialize wandb
#     wandb.finish()
#     wandb.init(
#         project=config.wandb_project,
#         config=vars(config)
#     )
#     run_id = wandb.run.id
    
#     # Load data
#     dataset = load_and_prepare_dataset(config)
    
#     # Setup model
#     model, tokenizer = setup_model(config)
    
#     # Preprocess
#     dataset_encoded = preprocess_dataset(dataset, tokenizer, config)
    
#     # Create training args
#     training_args = create_training_args(config, dataset_encoded, run_id)
    
#     # Data collator
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer,
#         padding=True
#     )
    
#     # Create metrics
#     compute_metrics = create_compute_metrics(tokenizer)
    
#     # Create trainer
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset_encoded["train"],
#         eval_dataset=dataset_encoded["valid"],
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#         callbacks=[
#             EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
#         ]
#     )
    
#         # Safety check
#     if not safe_training_check(trainer):
#         print("\n❌ Safety checks failed! Fix issues before training.")
#         return
    
#     # Train!
#     print("\n" + "=" * 60)
#     print("🏋️  Starting training...")
#     print("=" * 60)
    
#     try:
#         trainer.train()
#         print("\n✅ Training complete!")
#     except Exception as e:
#         print(f"\n❌ Training failed: {e}")
#         wandb.finish()
#         return
    
#     # Save model
#     best_model_path = f"{config.output_dir}/best_model"
#     save_model_safe(model, tokenizer, best_model_path, use_lora=config.use_lora)
    
#     print(f"\n🎉 All done! Model saved to {best_model_path}")
#     wandb.finish()


# if __name__ == "__main__":
#     main()
"""
Custom callbacks for training
"""
import torch
from transformers import TrainerCallback
import wandb

class SamplePredictionCallback(TrainerCallback):
    """Generate predictions on a few validation samples and log to W&B."""

    def __init__(self, tokenizer, eval_dataset, num_samples=5, max_length=64):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.max_length = max_length
        
    @torch.no_grad()
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation - generate sample predictions"""
        model = kwargs["model"]
        model.eval()
        device = model.device
        
        # Select sample rows
        samples = self.eval_dataset.select(range(min(self.num_samples, len(self.eval_dataset))))

        table = wandb.Table(columns=["Input", "Target", "Prediction", "Match"])

        for sample in samples:
            # Use the original raw text fields, not token IDs
            inp_text = sample["incorrect_sentence"]
            tgt_text = sample["correct_sentence"]
            
            # Tokenize individual sample
            tokenized = self.tokenizer(
                inp_text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=self.max_length
            ).to(device)
            
            # Safe generate (LoRA-friendly)
            with torch.cuda.amp.autocast(enabled=False):
                output_ids = model.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    max_length=self.max_length,
                    num_beams=1
                )
                
            pred_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Check if prediction matches target
            match = "✅" if pred_text.strip() == tgt_text.strip() else "❌"

            table.add_data(inp_text, tgt_text, pred_text, match)

            # Cleanup
            del tokenized, output_ids
            torch.cuda.empty_cache()
            
        # Log to wandb
        wandb.log({"sample_predictions": table, "epoch": state.epoch})
        
        print(f"📊 Logged {self.num_samples} sample predictions to W&B")
        
class MemoryCleanupCallback(TrainerCallback):
    """Clean up GPU memory periodically"""
    
    def on_step_end(self, args, state, control, **kwargs):
        """Clean memory every N steps"""
        if state.global_step % 100 == 0:
            torch.cuda.empty_cache()
from typing import List
from dataclasses import dataclass

@dataclass
class Config:
    """Single configuration class for everything"""
    
    # Model settings
    model_id: str = "google/mt5-small"
    max_length: int = 64
    prefix: str = "वाक्य सच्याउनुहोस्: "
    
    # Dataset settings
    dataset_name: str = "sumitaryal/nepali_grammatical_error_correction"
    num_samples: int = 15 # Set to None for full dataset
    valid_size: float = 0.1
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Set to None for all layers
    
    # Training settings
    batch_size: int = 16
    num_epochs: int  = 5
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    
    # Optimization settings
    use_8bit: bool = True
    use_fp16: bool = False   # Don't use with 8-bit
    gradient_checkpointing: bool = True
    
    # Logging & saving
    output_dir: str = "../outputs"
    logging_steps: int = 1
    save_total_limit: int = 2
    early_stopping_patience: int = 3
    
    # Wandb settings
    wandb_project: str = "nepali-grammar-correction"
    
    # Seeds
    seed: int = 42
    
    def __post_init__(self):
        """Set default values that depend on other attributes"""
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q", "k", "v", "o"]
            
    # Metrics configuratin -control which metric to compute
    bleu: bool = True
    chrf: bool = True
    gleu: bool = True
    correction_accuracy: bool = True
    bertscore: bool = False
    
    def get_enabled_metrics(self) -> List[str]:
        """Get list of enabled metrics"""
        enabled = []
        if self.bleu:
            enabled.append("bleu")
        if self.chrf:
            enabled.append("chrf")
        if self.gleu:
            enabled.append("gleu")
        if self.correction_accuracy:
            enabled.append("correction_accuracy")
        if self.bertscore:
            enabled.append("bertscore")
        return enabled
    
    # To continue training from latest checkpoint
    resume_from_checkpoint: bool = False
## Setup Instructions

1.  Clone this repository.

2.  The `data/` folder is ignored since it is too large. You need to create it yourself:
    ```bash
    mkdir data
    ```

3. Create file structure like below:

Recommend Project File Structure:

nepali_gec/
│
├── data/
│   ├── raw/                # Raw datasets (original files)
│   ├── inflected/          # error inflected datasets
│   │   ├── normal/
│   │   ├── multi/
│   │   └── semantic/
│   └── tiny_overfit/       # Small dataset for overfit test
│
├── adapters/               # LoRA adapters + fusion
│   ├── lora_easy/
│   ├── lora_multi/
│   ├── lora_semantic/
│   └── fusion_all/
│
├── notebooks/              # Jupyter / Colab notebooks
│   ├── overfit_test.ipynb
│   ├── train_lora_adapters.ipynb
│   ├── adapter_fusion.ipynb
│   └── evaluation.ipynb
│
├── src/                    # Python scripts for training / evaluation
│   ├── train_lora.py
│   ├── train_fusion.py
│   ├── dataset_utils.py
│   ├── metrics.py
│   └── inference.py
│
├── reports/                # Evaluation results, plots, logs
│   ├── fusion_eval.txt
│   └── loss_curves/
│
├── configs/                # Hyperparameters and experiment configs
│   ├── lora_config.json
│   └── training_args.json
│
├── requirements.txt        # Python dependencies
├── README.md               # Project description & instructions
└── .gitignore              # Ignore adapters, datasets if too big

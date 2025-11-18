## Setup Instructions

1.  Clone this repository.

2.  The `data/` folder is ignored since it is too large. You need to create it yourself:
    ```bash
    mkdir data
    ```

3. Create file structure like below:

Recommend Project File Structure:


nepali_gec/
├── README.md
├── data
│   ├── cleaned
│   ├── dictionary.txt
│   ├── inflected
│   ├── raw
│   └── readme.md
├── myenv
│   ├── Include
│   ├── Lib
│   ├── Scripts
│   ├── pyvenv.cfg
│   └── share
├── notebooks
│   ├── baseline.ipynb
│   ├── mt5_model.ipynb
│   └── mt5_with_lora.ipynb
├── outputs
│   ├── best_model
│   ├── checkpoints
│   └── logs
├── requirements.txt            # Python dependencies
├── src                          # Python scripts for training / evaluation
│   ├── __pycache__
│   ├── config.py               # Hyperparameters and experiment configs
│   ├── data_utils.py
│   ├── inference.py
│   ├── main.ipynb
│   ├── metrics.py
│   ├── train.py
│   ├── utils.py
│   └── wandb
└── wandb
    ├── run-20251117_232400-yiy0jf05
    └── run-20251117_232959-cxs01ivf
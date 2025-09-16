# all processes used to create 'clean_Nepal_text_corpus_ieee.txt'

"""
created by removing following things
1.  tags
2. special symbols
3. English
4. sentences with words < 3 and > 20
"""

import yaml
import pandas as pd
from pathlib import Path

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

raw_data_path = Path(config["raw_data"])
processed_data_path = Path(config["processed_data"])

# Use paths
df = pd.read_csv(raw_data_path)
print("Raw data loaded:", raw_data_path)

# Example: Save processed data
df.to_csv(processed_data_path, index=False)
print("Processed data saved:", processed_data_path)


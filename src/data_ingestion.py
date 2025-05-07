import os
import pandas as pd
# (your existing imports remain the same)

# Read raw data
df = pd.read_csv('./data/raw/train.txt', header=None, sep=";", names=['text', 'emotion'])
df = df.reset_index(drop=True)

# Ensure output directory exists
output_dir = './data/preprocess'
os.makedirs(output_dir, exist_ok=True)

# Save the processed file
df.to_csv(f'{output_dir}/train.csv', index=False)
print("Successfully saved CSV")

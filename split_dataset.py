import pandas as pd
import os

input_file = "hou_all.csv"

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)  

num_clients = 3

try:
    data = pd.read_csv(input_file)
    
    splits = []
    chunk_size = len(data) // num_clients
    for i in range(num_clients):
        if i < num_clients - 1:
            splits.append(data.iloc[i * chunk_size: (i + 1) * chunk_size])
        else:
            splits.append(data.iloc[i * chunk_size:])

    for idx, split in enumerate(splits, 1):
        output_file = os.path.join(output_dir, f"client_{idx}_data.csv")
        split.to_csv(output_file, index=False)
        print(f"Saved dataset for client_{idx} to {output_file}")

except Exception as e:
    print(f"Error: {e}")

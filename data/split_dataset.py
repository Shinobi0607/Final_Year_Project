import pandas as pd
import os

# Path to the main dataset
input_file = "hou_all.csv"

# Output directory for client-specific datasets
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Number of clients
num_clients = 3

try:
    # Read the dataset
    data = pd.read_csv(input_file)
    
    # Split the data into equal parts
    splits = []
    chunk_size = len(data) // num_clients
    for i in range(num_clients):
        if i < num_clients - 1:
            splits.append(data.iloc[i * chunk_size: (i + 1) * chunk_size])
        else:
            # Add remaining data to the last client
            splits.append(data.iloc[i * chunk_size:])

    # Save each split to a CSV file
    for idx, split in enumerate(splits, 1):
        output_file = os.path.join(output_dir, f"client_{idx}_data.csv")
        split.to_csv(output_file, index=False)
        print(f"Saved dataset for client_{idx} to {output_file}")

except Exception as e:
    print(f"Error: {e}")

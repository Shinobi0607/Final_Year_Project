import pandas as pd
import numpy as np

file_path = "hou_all.csv"
sequence_length = 10

data = pd.read_csv(file_path)
data.fillna(0, inplace=True)

features = data.values

def create_sequences(data, sequence_length):
    inputs, targets = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length, :-1]  # Input sequence (all but last column)
        target = data[i + sequence_length, -1]  # Target (last column)
        inputs.append(seq)
        targets.append(target)
    return np.array(inputs), np.array(targets)

inputs, targets = create_sequences(features, sequence_length)
np.save("processed_inputs.npy", inputs)
np.save("processed_targets.npy", targets)

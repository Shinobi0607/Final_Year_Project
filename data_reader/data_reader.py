import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset file {dataset_path} was not found.")
    
    data = pd.read_csv(dataset_path)
    return data

def preprocess_data(data, label_column_index=-1):
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, label_column_index].values 
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y

def get_data(dataset_path, label_column_index=-1):
    data = load_data(dataset_path)
    X, y = preprocess_data(data, label_column_index)
    return X, y

def get_data_train_samples(dataset_path, test_size=0.2, label_column_index=-1):
    X, y = get_data(dataset_path, label_column_index)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    dataset_path = "hou_all.csv"  
    
    X_train, y_train, X_test, y_test = get_data_train_samples(dataset_path)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

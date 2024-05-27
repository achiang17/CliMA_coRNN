import torch
import numpy as np
import pandas as pd

def parse_data(df, start, end, n=4):
    tensor_dict = {}
    for i in range(n):
        tensor_dict[i] = torch.tensor(df.iloc[start:end, i].values, dtype=torch.float32)
    data_seg = torch.stack([tensor_dict[i] for i in range(n)], dim=-1)
    return data_seg

def create_data(df, seq_len):
    data_dict = {}
    labels_dict = {}
    count = 0
    for i in range(0, len(df.index) - seq_len + 1, seq_len):
        data_dict[count] = parse_data(df, i, i + seq_len)
        labels_dict[count] = torch.tensor(df.iloc[i:i+seq_len, -1].values, dtype=torch.float32)
        count += 1
    data = torch.stack([data_dict[i] for i in range(count)], dim=1)
    labels = torch.stack([labels_dict[i] for i in range(count)], dim=0)
    return data, labels

def get_data(path_to_csv, seq_len):
    df = pd.read_csv(path_to_csv)
    df = df.transpose()
    data, labels = create_data(df, seq_len)
    return data, labels

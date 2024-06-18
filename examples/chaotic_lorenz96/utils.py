import torch
import numpy as np
import pandas as pd
import re

def parse_data(df, start, end, dim):
    tensor_dict = {}
    df_cols = list(df.columns.values)
    df_cols.remove(dim)
    for i,col in enumerate(df_cols):
        tensor_dict[i] = torch.tensor(df.iloc[start:end, col].values, dtype=torch.float32)
    data_seg = torch.stack([tensor_dict[i] for i in range(len(df_cols))], dim=-1)
    return data_seg

def create_data(df, seq_len, dim):
    data_dict = {}
    labels_dict = {}
    count = 0
    for i in range(0, len(df.index) - seq_len + 1, seq_len):
        data_dict[count] = parse_data(df, i, i + seq_len, dim)
        labels_dict[count] = torch.tensor(df.iloc[i:i+seq_len, dim].values, dtype=torch.float32)
        count += 1
    data = torch.stack([data_dict[i] for i in range(count)], dim=1)
    labels = torch.stack([labels_dict[i] for i in range(count)], dim=0)
    return data, labels

def get_data(path_to_csv, seq_len, dim, split, step):
    df = pd.read_csv(path_to_csv)
    df = df.transpose()

    pattern = r'/(\d+\.\d+)_\d+\.csv'
    match = re.search(pattern, path_to_csv)
    F = float(match.group(1))
    F_col = F * np.ones(len(df.index))
    df[5] = F_col

    if split == 'time':
        if step == 'train':
            df = df.iloc[:1000]
        else: # step = 'test'
            df = df.iloc[1000:]
            
    data, labels = create_data(df, seq_len, dim)
    return data, labels
import torch
import pandas as pd

def parse_data(df, start, end, n=5):
    tensor_dict = {}
    for i in range(n):
        tensor_dict[i] = torch.tensor(df.iloc[start:end, i].values, dtype=torch.float32)
    data_seg = torch.stack([tensor_dict[i] for i in range(n)], dim=-1)
    return data_seg

def create_data(df, traj_len):
    data_dict = {}
    count = 0
    for i in range(0, len(df.index) - traj_len + 1, traj_len):
        data_dict[count] = parse_data(df, i, i + traj_len)
        count += 1
    data = torch.stack([data_dict[i] for i in range(count)], dim=1)
    return data

def get_data(start, end, traj_len):
    frames = [pd.read_csv(f'train/train_{i}.csv').transpose() for i in range(start, end)]
    result = pd.concat(frames)
    data = create_data(result, traj_len)
    return data

def get_label(start,end):
    df = pd.read_csv('train/labels.csv')
    label = torch.tensor(df['F_vals'].iloc[start-1:end-1].values, dtype=torch.float32)
    label = label.unsqueeze(1)
    return label
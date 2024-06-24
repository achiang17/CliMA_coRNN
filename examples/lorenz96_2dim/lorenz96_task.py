import torch
from torch import nn, optim
import model as model_module
import torch.nn.utils
import utils
import argparse
import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from nse import NSELoss
from math import sqrt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=64, help='hidden size of recurrent net')
parser.add_argument('--seq_len', type=int, default=250, help='length of each sequence')
parser.add_argument('--dim', type=int, default=4, help='which dimension to predict from [0, 1, 2, 3, 4]')
parser.add_argument('--hdim', type=int, default=3, help='which dimension to predict from [0, 1, 2, 3, 4]')
parser.add_argument('--log_interval', type=int, default=100, help='log interval')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--dt', type=float, default=1e-2, help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=66, help='y control parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=15, help='z control parameter <epsilon> of the coRNN')

args = parser.parse_args()

n_inp = 4
n_out = 1

def initialize_model():
    return model_module.coRNN(n_inp, args.n_hid, n_out, args.dt, args.gamma, args.epsilon).to(device)

#objective = NSELoss()  # nn.MSELoss()

def test(model, path_to_test_csv, split, objective):
    model.eval()
    with torch.no_grad():
        data, label = utils.get_data(path_to_test_csv, args.seq_len, args.dim, args.hdim, split, "test")
        out = model(data.to(device))
        loss = objective(out[1:], label[1:].to(device))
    
    loss = loss.item()
    if isinstance(objective, NSELoss):
        loss = 1 - loss
    return loss

def predict(model, path_to_test_csv, split, train_objective, test_objective):
    model.eval()
    pattern = r'/(\d+\.\d+)_\d+\.csv'
    match = re.search(pattern, path_to_test_csv)
    F = float(match.group(1))
    train_loss_func = get_loss_from_obj(train_objective)   
    test_loss_func = get_loss_from_obj(test_objective)
    traj_len = 2000

    with torch.no_grad():
        if split == 'time':
            traj_len = 1000
        data, label = utils.get_data(path_to_test_csv, traj_len, args.dim, args.hdim, split, "test")
        out = model(data.to(device))
        out_np = out.cpu().numpy()
        label_np = torch.squeeze(label).cpu().numpy()

        plt.figure()
        plt.plot(out_np, label='Predicted')
        plt.plot(label_np, label='True')
        plt.title(f"{split.title()}-Split: Predicted vs Observed Trajectory F = {F}")
        plt.xlabel('Step')
        plt.ylabel('x4')
        plt.legend()
        plt.savefig(f'plots/{test_loss_func}/{split.title()}_{train_loss_func}_{test_loss_func}_traj.png')
    return

def train(model, optimizer, split, train_objective, test_objective):
    # random.seed(1)
    if split == "basin":
        train_dir = 'basin_split_train/'
        train_files = os.listdir(train_dir)
        random.shuffle(train_files)

        test_dir = 'basin_split_test/'
        test_files = os.listdir(test_dir)
        random.shuffle(test_files)
    elif split == "time":
        train_dir = "time_split"
        train_files = os.listdir(train_dir)
        random.shuffle(train_files)

        test_dir = train_dir
        test_files = train_files
        random.shuffle(test_files)
    else:
        raise ValueError(f"{split} is an invalid input. Instead use 'basin' or 'time'")

    test_err = []
    steps = []
    for i in range(len(train_files)):
        data, label = utils.get_data(os.path.join(train_dir, train_files[i]), args.seq_len, args.dim, args.hdim, split, "train")
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = train_objective(out, label.to(device))
        loss.backward()
        optimizer.step()

        if i % 1 == 0 and i != 0:
            test_csv_path = os.path.join(test_dir, test_files[i])

            # Plot Predicted vs True Trajectory
            if i == len(train_files) - 1:
                predict(model, test_csv_path, split, train_objective, test_objective)
            
            error = test(model, test_csv_path, split, test_objective)
            steps.append(i)
            test_err.append(error)
            model.train()

    return steps, test_err


def create_loss_plot(steps, test_err, split, train_objective, test_objective):
    train_loss_func = get_loss_from_obj(train_objective)   
    test_loss_func = get_loss_from_obj(test_objective)

    plt.figure()
    plt.plot(steps, test_err)
    plt.ylabel(test_loss_func)
    plt.xlabel("Training Steps")
    plt.title(f"{split.title()}-Split: {test_loss_func} Loss Trained with {train_loss_func} Loss")
    if test_loss_func.upper() == "NSE":
        count_lt = [err for err in test_err if err < 0]
        percent_NSE_lt = 100 * (len(count_lt) / len(test_err))
        print(f"{split.title()}-Split %NSE < 0 : {round(percent_NSE_lt, 2)}%")
        plt.ylim(-2, 1)
        plt.fill_between(steps, -2, 0, alpha=0.3)
        plt.text(0.7 * len(steps), 0.1, f"%NSE < 0 : {round(percent_NSE_lt, 2)}%")
    plt.savefig(f"plots/{test_loss_func}/{split.title()}_{train_loss_func}_{test_loss_func}_errors.png")


def get_cdf(test_err):
    test_err.sort()
    NSE_series = pd.Series(test_err)
    cdf_series = NSE_series.rank(method='first', pct=True)
    cdf_list = cdf_series.tolist()
    return test_err, cdf_list


def get_loss_from_obj(objective):
    if isinstance(objective, NSELoss):
        return "NSE"
    elif isinstance(objective, nn.MSELoss):
        return "MSE"
    else:
        return None

if __name__ == '__main__':
    split_vals = ["basin", "time"]
    train_objective_vals = [nn.MSELoss(), NSELoss()]
    test_objective = NSELoss()

    test_errs = {}
    for split in split_vals:
        for train_objective in train_objective_vals:
            train_loss_func = get_loss_from_obj(train_objective)
            model = initialize_model()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            epochs = 2
            for i in range(epochs):
                steps, test_err = train(model, optimizer, split.lower(), train_objective, test_objective)
            create_loss_plot(steps, test_err, split, train_objective, test_objective)
            test_errs[(split, train_loss_func)] = test_err


    # Additional CDF Plots for NSE loss
    if isinstance(test_objective, NSELoss):
        NSE_map = {}
        cdf_map = {}
        for (split, train_loss_func), error in test_errs.items():
            NSE, cdf = get_cdf(error)
            NSE_map[(split, train_loss_func)] = NSE
            cdf_map[(split, train_loss_func)] = cdf

        plt.figure(figsize=(6, 6))
        for (split, train_loss_func) in NSE_map.keys():     
            plt.plot(NSE_map[(split, train_loss_func)], cdf_map[(split, train_loss_func)], label=f'{split}: {train_loss_func} trained')
        plt.xlabel('NSE')
        plt.ylabel('CDF')
        plt.title('CDFs of NSE')
        plt.xlim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.savefig(f'plots/NSE/NSE_CDFs.png')
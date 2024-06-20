import torch
from torch import nn, optim
import model
import torch.nn.utils
import utils
import argparse
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from nnse import NNSELoss
from nkge import NKGELoss
from math import sqrt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=64,
    help='hidden size of recurrent net')
parser.add_argument('--seq_len', type=int, default=250,
    help='length of each sequence')
parser.add_argument('--dim', type=int, default=4,
    help='which dimension to predict from [0, 1, 2, 3, 4]')
parser.add_argument('--log_interval', type=int, default=100,
    help='log interval')
parser.add_argument('--lr', type=float, default=2e-2,
    help='learning rate')
parser.add_argument('--dt',type=float, default=1e-2,
    help='step size <dt> of the coRNN')
parser.add_argument('--gamma',type=float, default=66,
    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon',type=float, default = 15,
    help='z controle parameter <epsilon> of the coRNN')

args = parser.parse_args()

n_inp = 5
n_out = 1

model = model.coRNN(n_inp, args.n_hid, n_out, args.dt, args.gamma, args.epsilon).to(device)


objective = NKGELoss() # nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(path_to_test_csv, split, loss_func):
    model.eval()
    with torch.no_grad():
        data, label = utils.get_data(path_to_test_csv, args.seq_len, args.dim, split, "test")
        out = model(data.to(device))
        loss = objective(out[1:], label[1:].to(device))
    loss = loss.item()
    if loss_func == ("NNSE" or "NKGE"):
        loss = 1 - loss
    return loss

def predict(path_to_test_csv, split, loss_func):
    model.eval()
    
    pattern = r'/(\d+\.\d+)_\d+\.csv'
    match = re.search(pattern, path_to_test_csv)
    F = float(match.group(1))
    traj_len = 2000

    with torch.no_grad():
        if split == 'time':
            traj_len = 1000
        data, label = utils.get_data(path_to_test_csv, traj_len, args.dim, split, "test")
        out = model(data.to(device))
        out_np = out.numpy()
        label_np = torch.squeeze(label).numpy()

        plt.figure()
        plt.plot(out_np, label='Predicted')
        plt.plot(label_np, label='True')
        plt.title(f'{split.title()}-Split w/ {loss_func} loss Predicted vs True Trajectories: F = {F}')
        plt.xlabel('Step')
        plt.ylabel('x4')
        plt.legend()
        plt.savefig(f'plots/{loss_func}/{split.title()}_{loss_func}_Predicted_vs_Observed_Trajectory.png')
    return


def train(split, loss_func):
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
        data, label = utils.get_data(os.path.join(train_dir,train_files[i]), args.seq_len, args.dim, split, "train")
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = objective(out, label.to(device))
        loss.backward()
        optimizer.step()

        if(i%1==0 and i!=0):
            test_csv_path = os.path.join(test_dir,test_files[i])
           
            # predict
            if i == len(train_files)-1:
                predict(test_csv_path, split, loss_func)
            error = test(test_csv_path, split, loss_func)
            steps.append(i)
            test_err.append(error)
            model.train()

    return steps,test_err


if __name__ == '__main__':
    # time or basin
    split = "basin"
    # MSE, NNSE, or NKGE
    loss_func = 'NKGE'

    steps,test_err = train(split.lower(), loss_func.upper())

    plt.figure()
    plt.plot(steps,test_err)
    plt.ylabel(loss_func.upper())
    plt.xlabel("Training steps")
    plt.title(f"{split.title()}-Split Lorenz96: {loss_func} vs Training steps")
    if loss_func.upper() == "NNSE":
        count_lt = [err for err in test_err if err < 0.5]
        percent_NSE_lt = 100 * (len(count_lt) / len(test_err))
        print(f"%NSE < 0 : {round(percent_NSE_lt,2)}%")
        plt.ylim(0,1)
        plt.fill_between(steps, 0, 0.5, alpha=0.3)
        plt.text(0.7 * len(steps), 0.1, f"%NSE < 0 : {round(percent_NSE_lt_0,2)}%")
    if loss_func.upper() == "NKGE":
        count_lt = [err for err in test_err if err < 1-sqrt(2)]
        percent_KGE = 100 * (len(count_lt) / len(test_err))
        print(f"%KGE < 1 - sqrt(2) : {round(percent_KGE,2)}%")
        # plt.ylim(-5,1)
        # plt.fill_between(steps, -5, 1 - sqrt(2), alpha=0.3)
    
    plt.savefig(f"plots/{loss_func.upper()}/{split.title()}_{loss_func}_vs_steps_lorenz.png")
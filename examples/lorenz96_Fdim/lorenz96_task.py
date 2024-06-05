import torch
from torch import nn, optim
import model
#import nrmse
import torch.nn.utils
import utils
import argparse
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=128,
    help='hidden size of recurrent net')
parser.add_argument('--seq_len', type=int, default=100,
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


objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(path_to_test_csv, last_iter_bool):
    model.eval()
    with torch.no_grad():
        data, label = utils.get_data(path_to_test_csv, args.seq_len, args.dim)
        out = model(data.to(device))
        loss = objective(out, label.to(device))       
        if last_iter_bool:
            out_np = out.numpy()
            concat_out = np.concatenate(out_np).tolist()
            label_np = label.numpy()
            concat_label = np.concatenate(label_np).tolist()

            plt.figure()
            plt.plot(concat_out, label='Predicted')
            plt.plot(concat_label, label='True')
            plt.title('Predicted vs True Trajectories (x4)')
            plt.xlabel('Step')
            plt.ylabel('x4')
            plt.legend()
            plt.savefig('Predicted_vs_Observed_Trajectory.png')
    return loss.item()


def train():
    train_dir = 'train/'
    train_files = os.listdir(train_dir)
    random.shuffle(train_files)

    test_dir = 'test/'
    test_files = os.listdir(test_dir)
    random.shuffle(test_files)

    test_err = []
    steps = []
    last_iter_bool = False
    for i in range(len(train_files)):
        data, label = utils.get_data(os.path.join(train_dir,train_files[i]), args.seq_len, args.dim)
        optimizer.zero_grad()
        out = model(data.to(device))
        # print(f"out size:{out.size()}")
        # print(f"label size: {label.size()}")
        loss = objective(out, label.to(device))
        loss.backward()
        optimizer.step()

        if(i%1==0 and i!=0):
            if i == len(train_files)-1:
                last_iter_bool = True
            test_csv_path = os.path.join(test_dir,test_files[i])
            error = test(test_csv_path, last_iter_bool)
            steps.append(i)
            test_err.append(error)
            model.train()

    return steps,test_err


if __name__ == '__main__':
    steps,test_err = train()
    # print(f"test_mse: {mean(test_err[100:])}")
    plt.figure()
    plt.plot(steps,test_err)
    plt.ylabel("MSE")
    plt.xlabel("Training steps")
    plt.title(f"lorenz96: MSE vs Training steps")
    plt.savefig(f"MSE_vs_steps_lorenz.png")
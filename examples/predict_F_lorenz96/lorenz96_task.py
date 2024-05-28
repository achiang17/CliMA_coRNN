import torch
from torch import nn, optim
import model
import torch.nn.utils
import utils
import argparse
import os
import matplotlib.pyplot as plt
from statistics import mean

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=128,
    help='hidden size of recurrent net')
parser.add_argument('--traj_len', type=int, default=2001,
    help='number of trajectories to train per iteration')
parser.add_argument('--batch_size', type=int, default=10,
    help='number of trajectories to train per iteration')
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

def test(start, stop):
    model.eval()
    with torch.no_grad():
        data = utils.get_data(start, stop, args.traj_len)
        label = utils.get_label(start, stop)
        out = model(data.to(device))
        # print(f"true: {label}")
        # print(f"pred: {out}")
        loss = objective(out, label.to(device))    
    return loss.item()


def train():
    train_dir = f'train/'
    files = os.listdir(train_dir)
    num_files = len(files)-1

    test_err = []
    steps = []
    count = 0
    
    for i in range(1,num_files+1 - args.batch_size, args.batch_size):
        data = utils.get_data(i, i+args.batch_size, args.traj_len)
        label = utils.get_label(i, i+args.batch_size)
        # print(f"data: {data}")
        # print(f"data.size(): {data.size()}")
        # print(f"label: {label}")
        # print(f"label.size(): {label.size()}")
        optimizer.zero_grad()
        out = model(data.to(device))
        # print(f"train true: {label}")
        # print(f"pred: {out}")
        loss = objective(out, label.to(device))
        loss.backward()
        optimizer.step()
        count += 1
        if(count%1==0 and count!=0):
            error = test(i, i+args.batch_size)
            steps.append(count)
            test_err.append(error)
            model.train()

    return steps,test_err


if __name__ == '__main__':
    steps,test_err = train()
    # for param in model.parameters():
    #     print("new param \n")
    #     print(f"param.size : {param.size()}")
    #     print(param)
    #     print("\n\n")
    #print(f"test_mse: {test_mse}")
    #print(mean(test_err[100:]))
    plt.plot(steps,test_err)
    plt.title("lorenz96: MSE vs Training steps")
    plt.ylabel("MSE")
    plt.xlabel("Training steps")
    plt.savefig("MSE_vs_steps_lorenz.png")
import torch
from torch import nn, optim
import model
import torch.nn.utils
import utils
import argparse
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=128,
    help='hidden size of recurrent net')
parser.add_argument('--T_in', type=int, default=175,
    help='trajectories used to train')
parser.add_argument('--T_pred', type=int, default=25,
    help='trajectory to predict')
# parser.add_argument('--max_steps', type=int, default=2000,
#   help='max learning steps')
parser.add_argument('--log_interval', type=int, default=100,
    help='log interval')
# parser.add_argument('--batch', type=int, default=25,
#   help='batch size')
# parser.add_argument('--batch_test', type=int, default=25,
#   help='size of test set')
parser.add_argument('--lr', type=float, default=2e-2,
    help='learning rate')
parser.add_argument('--dt',type=float, default=6e-2,
    help='step size <dt> of the coRNN')
parser.add_argument('--gamma',type=float, default=66,
    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon',type=float, default = 15,
    help='z controle parameter <epsilon> of the coRNN')

args = parser.parse_args()

n_inp = 5
n_out = 5

model = model.coRNN(n_inp, args.n_hid, n_out, args.dt, args.gamma, args.epsilon).to(device)

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(path_to_test_csv):
    model.eval()
    with torch.no_grad():
        data, label = utils.get_data(path_to_test_csv, args.T_in, args.T_pred)
        #label = label.unsqueeze(1)
        out = model(data.to(device))
        loss = objective(out, label.to(device))
    return loss.item()


def train():
    train_dir = 'train/'
    files = os.listdir(train_dir)
    num_files = len(files)

    test_mse = []
    steps = []
    
    for i in range(1,num_files+1):
        data, label = utils.get_data(f"train/train_{i}.csv", args.T_in, args.T_pred)
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = objective(out, label.to(device))
        loss.backward()
        optimizer.step()

        if(i%1==0 and i!=0):
            mse_error = test(f"test/test_{i}.csv")
            print('Test MSE: {:.6f}'.format(mse_error))
            steps.append(i)
            test_mse.append(mse_error)
            model.train()

    return steps,test_mse


if __name__ == '__main__':
    steps,test_mse = train()
    #print(f"test_mse: {test_mse}")
    plt.plot(steps,test_mse)
    plt.title("lorenz96: MSE vs Training steps")
    plt.ylabel("MSE")
    plt.xlabel("Training steps")
    plt.savefig("MSE_vs_steps_lorenz.png")
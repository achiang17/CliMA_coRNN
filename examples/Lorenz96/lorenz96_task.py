import torch
from torch import nn, optim
import model
import nrmse
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
parser.add_argument('--dt',type=float, default=1e-2,
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
        out = model(data.to(device))
        loss = objective(out, label.to(device))
        rmse_loss = torch.sqrt(loss)
        rms_true = torch.sqrt(torch.mean(label ** 2))
        nrmse_loss = rmse_loss/rms_true
        
    return loss.item()


def train():
    train_dir = 'train8/'
    files = os.listdir(train_dir)
    num_files = len(files)

    test_err = []
    steps = []
    
    for i in range(1,num_files+1):
        data, label = utils.get_data(f"train8/train8_{i}.csv", args.T_in, args.T_pred)
        # print(f"data: {data}")
        # print(f"data.size(): {data.size()}")
        # print(f"label: {label}")
        # print(f"label.size(): {label.size()}")
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = objective(out, label.to(device))
        rmse_loss = torch.sqrt(loss)
        rms_true = torch.sqrt(torch.mean(label ** 2))
        nrmse_loss = rmse_loss/rms_true
        loss.backward()
        optimizer.step()

        if(i%1==0 and i!=0):
            error = test(f"test8/test8_{i}.csv")
            #print('Test NRMSE: {:.6f}'.format(mse_error))
            steps.append(i)
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
    print(mean(test_err[100:]))
    plt.plot(steps,test_err)
    plt.title("lorenz96: MSE vs Training steps, F = 8.0")
    plt.ylabel("MSE")
    plt.xlabel("Training steps")
    plt.savefig("MSE_vs_steps_lorenz8.png")
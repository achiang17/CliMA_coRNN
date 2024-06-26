from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self, x, hy, hz):      
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy), 1)))
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz
        return hy, hz

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.cell = coRNNCell(n_inp, n_hid, dt, gamma, epsilon)
        self.readout = nn.Linear(n_hid, n_out)
        
    def forward(self, x):
        hy = Variable(torch.zeros(x.size(1), self.n_hid, dtype=torch.float32)).to(device)
        hz = Variable(torch.zeros(x.size(1), self.n_hid, dtype=torch.float32)).to(device)
        output = torch.zeros(x.size(1), x.size(0), self.n_out, requires_grad=False)
        for t in range(x.size(0)):
            hy, hz = self.cell(x[t], hy, hz)
            output[:,t,:] = self.readout(hy)
        output = torch.squeeze(output)
        return output

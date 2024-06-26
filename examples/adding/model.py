from torch import nn
import torch
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self,x,hy,hz):
        concatenated = torch.cat((x, hz, hy), 1)
        
        # Debug statements
        print(f"x shape: {x.shape}")
        print(f"hz shape: {hz.shape}")
        print(f"hy shape: {hy.shape}")
        print(f"concatenated shape: {concatenated.shape}")
        print(f"i2h weight shape: {self.i2h.weight.shape}")
        
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy),1)))
                                   - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp,n_hid,dt,gamma,epsilon)
        self.readout = nn.Linear(n_hid, n_out)

    def forward(self, x):
        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        print(f"hy size : {hy.size()}")
        for t in range(x.size(0)):
            hy, hz = self.cell(x[t],hy,hz)
        output = self.readout(hy)
        print(f"output: {output}")
        print(f"forard output size: {output.size()}")
        print(f"hy: {hy}")
        return output
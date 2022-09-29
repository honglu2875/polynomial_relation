import torch
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torch import optim

def g1(x, y):
    return x * y

def g2(x, y):
    return x * y

def g3(x, y):
    return torch.zeros_like(x)


class SmallNet(nn.Module):
    def __init__(self, input_dim, fns):
        super().__init__()
        self.input_dim = input_dim
        self.fns = fns
        self.lin1 = nn.Linear(len(self.fns), len(self.fns), bias=True)
        self.output = nn.Linear(len(self.fns), 1, bias=True)

    def forward(self, x):
        out = []
        inp_as_args = [x[:, i:i+1] for i in range(self.input_dim)]
        for f in self.fns:
            out.append(f(*inp_as_args))
        x = torch.cat(out, 1)  # Pass through the given functions
        y = self.lin1(x)  
        y = y * y  # Activation function 
        out = self.output(y)
        return out


def recursive_reg(x):
    if isinstance(x, dict):
        s = 0
        minimal = 100000
        for value in x.values():
            #s += torch.sum(torch.abs(recursive_reg(value)))
            minimal = min(torch.min(torch.abs(recursive_reg(value))), minimal)
        return minimal
    #return torch.sum(torch.abs(x))
    return torch.min(torch.abs(x))

def main():
    epochs = 1000
    input_dim = 2
    sample_size = 100
    fns = (g1, g2, g3)

    summary = SummaryWriter(log_dir="runs")

    net_ground_truth = SmallNet(input_dim, fns)
   
    """ 
    # The ground truth network for testing purpose
    net_ground_truth = SmallNet(input_dim, fns)
    state_dict = net_ground_truth.state_dict()
    state_dict['lin1.weight'] = torch.tensor([[1, 0, 0], [0, 1, 1], [0, 1, -1]]) + torch.rand((3,3)) * 0.5
    state_dict['lin1.bias'] = torch.zeros((3,))
    state_dict['output.weight'] = torch.tensor([[1, -0.25, 0.25]])
    state_dict['output.bias'] = torch.tensor([0])
    #state_dict['output.bias'] = torch.tensor([0])
    net_ground_truth.load_state_dict(state_dict)
    """

    opt = optim.Adam(net_ground_truth.parameters(), lr=0.001)
    loss_fn = nn.functional.mse_loss
    zero = torch.zeros((sample_size, 1))
    for i in range(epochs):
        if loss>0.02:
         opt.lr=0.02   
        elif loss>0.01:
         opt.lr=0.01
        elif loss>0.001:
         opt.lr=0.005
        elif loss>0.0002:
         opt.lr=0.003   
        else:
         opt.lr=0.0015
        
        inp = torch.rand((sample_size, input_dim)) * 1
        #opt.lr=0.001/(i+1)
        
        # Training
        opt.zero_grad()
        output = net_ground_truth(inp)
        #loss = loss_fn(output, zero) / recursive_reg(net_ground_truth.state_dict()) ** 4
        #print(net_ground_truth.state_dict())
        #print(recursive_reg(net_ground_truth.state_dict()))

        #loss = loss_fn(output, zero) / (torch.sum(torch.abs(state_dict['lin1.weight'])**4) * torch.sum(torch.abs(state_dict['output.weight']))**2)
        loss = loss_fn(output, zero) 

        loss.backward()
        
        opt.step()
        
        state_dict = net_ground_truth.state_dict()
        
        if i%100==0:
            print(loss)
            summary.add_scalar('loss', loss, i)
            for key, value in state_dict.items():
                summary.add_histogram(key, value, i)

    for i in range(10):
        inp = torch.rand((sample_size, input_dim)) * 10
        #print(net_ground_truth(inp))
        print(loss_fn(net_ground_truth(inp), zero))
    
    print(net_ground_truth.state_dict())


if __name__ == '__main__':
    main()


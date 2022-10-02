import torch
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torch import optim

def g1(x, y):
    return x * y

def g2(x, y):
    return x * x

def g3(x, y):
    return y * y
    #return torch.zeros_like(x)


class SmallNet(nn.Module):
    def __init__(self, input_dim, fns):
        super().__init__()
        self.input_dim = input_dim
        self.fns = fns
        self.lin1 = nn.Linear(len(self.fns), len(self.fns), bias=False)
        self.output = nn.Linear(len(self.fns), 1, bias=False)

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
    epochs = 50000
    input_dim = 2
    sample_size = 100
    fns = (g1, g2, g3)

    summary = SummaryWriter(log_dir="runs")

    net = SmallNet(input_dim, fns)
   
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

    loss_fn = nn.functional.mse_loss
    zero = torch.zeros((sample_size, 1))
    loss = 100
    state_dict = net.state_dict()
    state_dict['lin1.weight'][0, 0] = 1.0
    #state_dict['lin1.weight'][0, 0].requires_grad = False
    opt = optim.Adam(net.parameters(), lr=0.001)
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
        output = net(inp)
        #loss = loss_fn(output, zero) / recursive_reg(net.state_dict()) ** 4
        #print(net.state_dict())
        #print(recursive_reg(net.state_dict()))

        #loss = loss_fn(output, zero) / (torch.sum(torch.abs(state_dict['lin1.weight'])**4) * torch.sum(torch.abs(state_dict['output.weight']))**2)
        loss = loss_fn(output, zero) 

        loss.backward()
        
        opt.step()
        #print(state_dict['lin1.weight'][0, 0].requires_grad)
        state_dict['lin1.weight'][0, 0] = 1.0
        state_dict['output.weight'][0, 0] = 1.0
        
        
        
        if i%1000==0:
            print(loss)
            #summary.add_scalar('loss', loss, i)
            #for key, value in state_dict.items():
            #    summary.add_histogram(key, value, i)
        
    for i in range(10):
        inp = torch.rand((sample_size, input_dim)) * 10
        #print(net(inp))
        print(loss_fn(net(inp), zero))

    print("------")
    print(net(torch.tensor([[1.0, 0.0]])))
    print(net(torch.tensor([[0.0, 1.0]])))
    
    
    print(net.state_dict())


if __name__ == '__main__':
    main()


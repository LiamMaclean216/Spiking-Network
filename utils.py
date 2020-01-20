import torch
import math

def sig(x):
    return (1 / (1 + torch.exp(-x)))

def sig_d(x):
    s = sig(x)
    return s * (1 - s)



class Dense(torch.nn.Module):
    def __init__(self, n_in, n_out , sigmoid = False, pointwise = False):
        super(Dense, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.sigmoid = sigmoid
        self.pointwise = pointwise
        
        self.weight = torch.ones([n_in, n_out])
            
    
       # self.weight += -1
        self.reset_parameters()

    def forward(self, spikes):
        a = torch.zeros([self.n_out])
        for r, c in zip(spikes,self.weight):
            if self.sigmoid:
                a += sig(r * c) * r
            else:
                a += (r * c)
        
        #print(self.weight)
        return a

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(0, stdv)
        
    def weight_update(self, error, lr = 1):
        delta_w = sig_d(self.weight) * sig_d(error/5) * torch.sign(error) * lr
        self.weight += delta_w    
        return (sig(self.weight))

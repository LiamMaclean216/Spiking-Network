import torch
import math
import numpy as np

def normal(x, m, s):
    a = 1/torch.sqrt(2*math.pi*(s**2))
    return ( a* torch.exp(-((x-m)**2)/(2*s**2)))

class Node(torch.nn.Module):
    def __init__(self, thresh = 1, decay_rate = 0.8, lr = 0.1, name = "Node"):
        super(Node, self).__init__()
        self.decay_rate = torch.tensor(decay_rate).float()
        self.lr = lr
        
        self.connections = []
        self.name = name
        
        self.thresh = thresh
        
    def add(self, x):
        self.connections.append(x)
        self.weights = torch.zeros(len(self.connections))
        self.charges = torch.zeros(len(self.connections))
        
        self.expected_error = torch.ones(len(self.connections)) #* 0.1
        self.mean =  torch.ones(len(self.connections)) * 0.5
        
    def get_ratios(self):
        uncertainties = torch.zeros([len(self.connections),2])
        for i,c in enumerate(self.connections):
            uncertainties[i][0] = self.get_uncertainty(self.connections[i])
            uncertainties[i][1] = self.connections[i].get_uncertainty(self)
        
        ratios = uncertainties[:,0]/uncertainties[:,1]
        
        return ratios
    
    def correct_weights(self):
        r = self.get_ratios()
        
        likelihoods = torch.zeros([len(self.connections),2])
        
        #collect likelihoods of all connections
        for i in range(len(self.connections)):
            likelihoods[i,0] = self.connections[i].get_maxl(self)
            likelihoods[i,1] = self.get_maxl_ind(i)
        
        #correction value
        c = (-likelihoods[:,0] - (r * likelihoods[:,1])) / (r + 1)
              
        #normilazing coefficient
        n = torch.max(likelihoods[:,0], likelihoods[:,1]).float()    
              
        out = (likelihoods[:,1] + c) / n
        return out
    
    def decay(self):
        self.charges *= self.decay_rate
        self.charges[self.charges < 0] = 0
    
    def forward(self, x):
        noise = torch.distributions.Normal(0,0.01).sample([len(self.connections)])
        self.charges += x + (noise * x)
        self.decay()
            
        return  torch.sum(x * self.weights)
    def in_spike(self):
        c = self.charges
        
        lr = 0.3
        
        self.expected_error += (((torch.abs(c - self.mean))-self.expected_error))*lr
        self.mean = c
        
        new_w = self.correct_weights()
        delta_w = (new_w - self.weights) #* lr
        
        self.weights += delta_w
        self.charges = torch.zeros(self.charges.shape)
    
    def get_uncertainty(self,x, c = -1):
        for idx,s in enumerate(self.connections):    
            if s is x:
                if(c == -1):
                    c = self.mean[idx]
                return self.get_uncertainty_ind(c, idx)
        raise Exception('Connecting Neuron not found')
    
    def get_uncertainty_ind(self, c, idx ):
        v = ((torch.log10(c + 0.0000001 ) 
              - (torch.log10(c + 0.0000001  + self.expected_error[idx])))/torch.log10(self.decay_rate))
        return v
    
    def get_maxl(self, x):
        for idx,c in enumerate(self.connections):   
            if c is x:
                return self.get_maxl_ind(idx)
        raise Exception('Connecting Neuron not found')
        
    def get_maxl_ind(self, idx):
        return normal(self.mean[idx] , self.mean[idx], self.get_uncertainty_ind(self.mean[idx], idx))
    
    def __str__(self):
        return ("Weights : {}, mean : {}, exp_error : {}".format(self.weights, self.mean, 
                                                                                   self.expected_error))


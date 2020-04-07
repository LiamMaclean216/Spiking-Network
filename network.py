import torch
from utils import *

class SpikingNeuron(torch.nn.Module):
    def __init__(self, connections, n_inputs, verbose = False, thresh = 1, lr = 0.1, name = "Neuron"):
        super(SpikingNeuron, self).__init__()
        self.thresh = thresh
        self.verbose = verbose
        self.lr = lr
        self.n_inputs = n_inputs
        self.name = name
        
        #Position for visualisation
        position = [0,0]
        self.spiking = False
        self.to_spike = False
        self.connections = connections
        
    
    def add_connection(self, x):
        self.connections.append(x)
        
    def init(self) :
        self.nodes = []
        for i in range(len(self.connections) + self.n_inputs):
            self.nodes.append(Node(thresh = self.thresh, name = "{}Node{}".format(self.name, i)))
            
        for i in self.nodes:
            for j in self.nodes:
                if not (i is j):
                    i.add(j)
                    
        
    def forward(self, inputs = torch.tensor([])):
        x = torch.zeros([len(self.connections)]).bool()
        for i in range(len(self.connections)):
            x[i] = self.connections[i].spiking
        
        x = torch.cat((inputs.bool(), x))
        
        self.spiking = self.to_spike
        self.to_spike = False
        
        in_spikes = x.squeeze().nonzero()
        
        
        
        for i in in_spikes:
            self.nodes[i].in_spike()
            if(self.verbose):
                print(self.nodes[i])
                print()
            
        for i, n in enumerate(self.nodes):
            ins = torch.cat((x[:i], x[i+1:]), dim = 0)
            if(n(ins) >= self.thresh and not x[i]):
                self.to_spike = True    
       
            
        return torch.tensor([self.spiking])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        neurons = []
        self.spike1 = SpikingNeuron([], n_inputs = 2, verbose = True, thresh = 0.7, name = "Neuron1")
        neurons.append(self.spike1)
        
        self.spike2 = SpikingNeuron([self.spike1], n_inputs = 1, verbose = False, thresh = 0.7,name = "Neuron2")
        self.spike2.init()
        neurons.append(self.spike2)
        
        self.spike1.add_connection(self.spike2)
        self.spike1.init()
        
        #self.spike3 = SpikingNeuron(2)
        #self.s3 = torch.tensor([0]).bool()
       
        self.neurons = neurons
        
    def forward(self, x, training = True):
        x = x.bool()
        
        self.s1 = self.spike1(x)
        
        self.s2 = self.spike2(x[1].unsqueeze(0))
        
        #self.s3, v = self.spike3(torch.stack((self.s1,self.s2)))
        #draw.append(v + self.s3)
        
        return self.neurons
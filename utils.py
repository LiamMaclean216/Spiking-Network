import torch

#A block of every other spiking layer (updates them all at once)
class Spike1d_block(torch.nn.Module):
    def __init__(self, layers):
        super(Spike1d_block, self).__init__()
        self.layers = layers
        
    def forward(self,in_spikes):
        out_spikes = []
        out_pots = []
        for idx,l in enumerate(self.layers):
            o, p = l(in_spikes[idx:idx + l.get_n_inputs()])
            out_pots.append(p)
            out_spikes.append(o)
        
        return out_spikes, out_pots
    
    def get_empty_spikes(self):
        out = []
        for i in self.layers:
            out.append(i.get_empty_spikes())
        return out

#this function recombines and draws the lists of odd and even spikes
def draw_spikes(spikes, ax, l_dim = 4, n_layers = 3):
    to_draw = torch.zeros(n_layers,l_dim) - 0.1
    e = even = odd = 0
    while True:
        e += 1
        if e % 2 != 0:
            spike_block = spikes[0]
            i = odd
            odd += 1
        else:
            spike_block = spikes[1]
            i = even
            even += 1
        
        if(i > len(spike_block)-1):
            break
        offset = (l_dim - spike_block[i].shape[0]) // 2
        try:
            to_draw[e-1][offset:spike_block[i].shape[0] + offset] = spike_block[i].clone().detach()
        except:
            to_draw[e-1][offset:spike_block[i].shape[0] + offset] = torch.tensor(spike_block[i])

    to_draw = to_draw.t()
    
    ax.clear()
    ax.imshow(to_draw,vmin=-0.1, vmax=1)  
import torch
from torch import nn

from Convolution import Convolution


class ConvolutionLayer(nn.Module):
    def __init__(self, num_filters, filter_size, in_channels, connections=None):
        super().__init__()
        self.fc = not connections
        self.connections = connections
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.tanh = nn.Tanh()
        
        if self.fc:
            self.filters = nn.ModuleList([
                    Convolution(size=filter_size, in_channels=in_channels) 
                    for i in range(num_filters)
                    ])
        else:
            assert connections is not None, "in_channels and connections args can't both be None in conv layer" 
            self.filters = nn.ModuleList([
                Convolution(size=filter_size, in_channels=len(connections[i])) 
                for i in range(num_filters)
                ])
        
    def forward(self, x, config):
        
        amp = config['A']
        scale = config['S']
        efficient = config['efficient']
        
        output = None
        if efficient:
            futures = [torch.jit.fork(filter, self.get_input(i, x)) for i, filter in enumerate(self.filters)]
            outputs = [torch.jit.wait(fut) for fut in futures]
            output = torch.cat(outputs, dim=1)
        else:
            # iterate through filters, apply, and extract features
            for i, filter in enumerate(self.filters):
                
                # if fully connected to previous layer, send all channels to filter
                if self.fc:
                    convolution = filter(x)
                else:
                    convolution = filter(self.get_input(i, x))
                    
                if output is None:
                    output = convolution
                else:
                    output = torch.cat([output, convolution], dim=1)
                    
        assert torch.all(torch.isfinite(output)), f"output not fully populated at conv layer {self.num_filters}"
    
        # scale, apply activation function, and multiply by amplitude 
        return amp * self.tanh(scale * output)
    
    def get_input(self, i, x):
        
        # get list of inputs to filter i (e.g. [1, 2, 4, 5])
        input_list = self.connections[i]
        
        # create list containing input tensors for filter i using list comprehension
        inputs = [x[:, c:c+1,:,:] for c in input_list]
        
        # concatenate list into tensor
        return torch.cat(inputs, dim=1)
from torch import nn
import torch

from convolution import Convolution

class ConvolutionLayer(nn.Module):
    '''
    defines a class for a single convolution layer in the LeNet-5 model
    
    args:
    - num_filters: int
    - filter_size: int
    - in_channels: int
    '''
    
    def __init__(self, in_channels, num_filters, filter_size, efficient=False, connections=None):
        super().__init__()
        
        # initialize useful fields
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.efficient = efficient
        self.connections = connections  
        self.fc = not connections
        
        if self.fc:
            # if layer is fully connected, each filter's input channel number
            # equals this layer's total number of input channels (in_channels)
            self.filters = nn.ModuleList([
                    Convolution(size=filter_size, in_channels=in_channels) 
                    for i in range(num_filters)
                    ])
        else:
            # ensure valid arguments
            assert connections is not None, "in_channels and connections args can't both be None in conv layer" 
            
            # assign filters
            self.filters = nn.ModuleList([
                Convolution(size=filter_size, in_channels=len(connections[i])) 
                for i in range(num_filters)
                ])
        
    def forward(self, x):
        '''
        runs a set of feature maps from the previous layer through convolutions
        
        args:
        - x: tensor (batch_size, in_channel_num, in_size, in_size)
        
        output:
        - tensor (batch_size, num_filters, out_size, out_size)
        '''
        
        # get input dimensions
        batch_size, in_channels, in_size, in_size = x.shape
        
        # validate input channel size
        assert self.in_channels == in_channels, f'channel number mismatch in forward pass in conv layer {self.in_channels}'
        
        # calculate output_size
        out_size = in_size - self.filter_size + 1
        
        # initialize output tensor
        output = torch.empty(batch_size, self.num_filters, out_size, out_size, device=x.device)
        
        # iterate through filters, apply, and extract features (slow)
        for i, filter in enumerate(self.filters):
            
            # if fully connected to previous layer, send all channels to filter
            if self.fc:
                convolution = filter(x, self.efficient)
            else:
                # (batch_size, 1, out_size, out_size)
                convolution = filter(self._get_input(i, x), self.efficient)

            # asign local convolution to corresponding output region
            output[:, i, :, :] = convolution
        
        # ensure output is fully populated by convolutions
        assert not torch.isnan(output).any(), f"output not fully populated at conv layer {self.num_filters}"

        # remove all singleton dimensions (e.g. (5, 1, 1) -> (5))
        output = output.squeeze()
        
        # if single batch, add batch dimension back
        # (e.g. (5) -> (1, 5))
        if batch_size == 1:
            output = output.unsqueeze(0)
        return output
    
    def _get_input(self, i, x):
        
        # get list of inputs to filter i (e.g. [1, 2, 4, 5])
        input_list = self.connections[i]
        
        # use advanced indexing (passing in a list in slicing) to filter input
        return x[:, input_list, :, :]
    
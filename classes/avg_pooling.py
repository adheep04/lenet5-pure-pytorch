from torch import nn
import torch

class AvgPoolingLayer(nn.Module):
    '''
    an avg-pooling layer
    
    args:
    - num_channels: int
    '''
    def __init__(self, num_channels, efficient=False):
        super().__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.efficient = efficient
    
    def forward(self, x):
        '''
        performs an average pooling downsampling of input across last 2 dimensions
        
        args:
        - x: tensor (batch_size, num_channels, in_size, in_size)
        
        output:
        - (batch_size, num_channels, in_size // 2, in_size // 2)
        
        '''
        
        # get input dimensions
        batch_size, channels, in_size, in_size = x.shape
        
        # validate input dimensions
        assert channels == self.num_channels, f'input channel num and expected channel num mismatch in pooling layer {self.num_channels}'
        assert in_size % 2 == 0, "input size must be divisible by 2 for pooling."
        
        # define output size
        out_size = in_size//2
        
        # initialize output tensor
        output = torch.empty(batch_size, self.num_channels, out_size, out_size, device=x.device)
        
        if not self.efficient:
            for i in range(0, in_size, 2):
                for j in range(0, in_size, 2):
                    
                    # manually add the 2x2 range and divide by 4, getting an
                    # average value for the corresponding 1x1 value  
                    avg_val = (
                        x[:, :, i, j] + 
                        x[:, :, i+1, j] + 
                        x[:, :, i, j+1] + 
                        x[:, :, i+1, j+1]
                        ) / 4
                    
                    output[:,:, i//2, j//2] = avg_val
        else:
            # vectorized method, faster but less readable
            output = x.reshape(batch_size, channels, out_size, out_size, 2, 2).mean(dim=(4,5))
            
        # validate pooling tensor
        assert not torch.isnan(output).any(), f"output tensor not fully populated in pooling layer {self.num_channels}"
        
        
        # reshape from (num_channels,) to (1, num_channels, 1, 1) for dimensional compatability
        # with output
        weights = self.weight.reshape(1, self.num_channels, 1, 1)
        biases = self.bias.reshape(1, self.num_channels, 1, 1)

        # multiply weights by output and add bias
        # output: (b, c, s/2, s/2)
        return (weights * output) + biases

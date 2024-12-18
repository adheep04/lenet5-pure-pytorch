import torch
from torch import nn

class AvgPoolingLayer(nn.Module):
    '''
    an avg-pooling layer
    '''
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.weights = nn.Parameter(torch.ones(num_channels))
        self.biases = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x, efficient):
        batch_size, channels, in_size, in_size = x.shape
        assert channels == self.num_channels, f'input channel num and expected channel num mismatch in pooling layer {self.num_channels}'
        assert in_size % 2 == 0, "input size must be divisible by 2 for pooling."
        out_size = in_size//2
        
        output = torch.empty((batch_size, self.num_channels, out_size, out_size))
        
        if not efficient:
            for i in range(0, in_size, 2):
                for j in range(0, in_size, 2):  
                    avg_val = (
                        x[:, :, i, j] + 
                        x[:, :, i+1, j] + 
                        x[:, :, i, j+1] + 
                        x[:, :, i+1, j+1]
                        ) / 4
                    output[:,:, 
                            i//2:(i//2)+1,
                            j//2:(j//2)+1] = avg_val
        else:
            output = x.reshape(batch_size, channels, out_size, out_size, 2, 2).mean(dim=(4,5))
            
        assert torch.all(torch.isfinite(output)), f"output tensor not fully populated in pooling layer {self.num_channels}"
        
        # output: (b, c, s/2, s/2)   
        return (self.weights * output) + self.biases
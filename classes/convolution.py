from torch import nn
import torch

class Convolution(nn.Module):
    '''
    defines a class for a square convolution with a stride of 1.
    **this class is not generalized for all convolutions, its for LeNet-5's purposes
    
    args:
    - size -> int
    - in_channels -> int
    '''
    def __init__(self, in_channels, size,):
        super().__init__()
        # number of incoming feature maps
        self.in_channels = in_channels
        # size of convolution (e.g. 3: 3x3 convolution)
        self.size = size
        # kernel which will be applied to input patches
        self.weight = nn.Parameter(data=torch.ones(size=(self.in_channels, self.size, self.size)))
        # bias scalar
        self.bias = nn.Parameter(torch.zeros((1)))
        # fan-in value for the weights in the convolution 
        # (useful for weight initialization during training)
        self.fan_in = in_channels * size**2
    
    def forward(self, x, efficient=False):
        '''
        performs a convolution and produces a feature map given an input
        
        args:
        - x: tensor (batch_size, in_channels_num, in_size, in_size)
        
        output:
        - tensor (batch_size, out_size, out_size)
        '''
        
        # get dimensions
        batch_size, in_channels, input_size, input_size = x.shape
        
        # calculate output size
        output_size = input_size - self.size + 1
        
        # validate input channel number
        assert self.in_channels == in_channels, 'input channel size mismatch with accepted num of channels'
        
        ''' two versions of forward pass, one efficient (pytorch operations), one not (for loops)'''
        
        if efficient:
            # Convert input into patches all at once
            patches = x.unfold(2, self.size, 1).unfold(3, self.size, 1)
            
            # Reshape for batch matrix multiplication
            patches = patches.reshape(batch_size, in_channels, -1, self.size*self.size)
            
            # Reshape weight to match patch dimensions for multiplication
            weight = self.weight.reshape(in_channels, -1)  # reshape to (in_channels, kernel_size*kernel_size)
            
            # Compute convolution for all patches at once
            # (batch_size, in_channels, num_patches, kernel_size*kernel_size) * (in_channels, kernel_size*kernel_size)
            # Sum over channels and kernel dimensions
            output = torch.sum(patches * weight.reshape(1, in_channels, 1, -1), dim=(1, 3))
            
            # Reshape output to match original spatial dimensions
            output = output.reshape(batch_size, output_size, output_size)
        else:       
            # initialize output tensor in the same device as the input
            output = torch.empty(batch_size, output_size, output_size, device=x.device)
                
            ''' using for loops for readability and clarity'''
            # vertical "slide" of convolution filter
            for i in range(output_size):
                
                # horizontal "slide" of convolution filter
                for j in range(output_size):
                    
                    # get patches of input using slicing to apply convolution on
                    # i and j represent the top left pixel in the convolution patch
                    input_patches = x[:, :, i:i+self.size, j:j+self.size]
                    
                    # apply convolution and save result in output
                    # covolution: sum of hadmard product between input patch and kernel (dot product)
                    # (b,) = sum((b, c, 5, 5) * (c, 5, 5))
                    output[:, i, j] = torch.sum(input_patches * self.weight, dim=(1,2,3)) 
        
        # validate output tensor
        assert not torch.isnan(output).any(), f"output not fully populated at conv filter {self.in_channels}"
        
        # add bias
        # output: (batch_size, output_size, output_size)
        return output + self.bias
    
import torch
from torch import nn

class Convolution(nn.Module):
    '''
    defines a class for a square convolution with a stride of 1.
    **this class is not generalized for all convolutions, its for LeNet-5's purposes
    doens't use F.unfold
    
    args:
    - size -> int
    - in_channels -> int
    '''
    def __init__(self, size, in_channels):
        super().__init__()
        # size of convolution (e.g. 3: 3x3 convolution)
        self.size = size
        # number of incoming feature maps
        self.in_channels = in_channels
        # kernel which will be applied to input patches
        self.kernel = nn.Parameter(data=torch.rand(size=(1, self.in_channels, self.size, self.size)))
        # bias scalar
        self.bias = nn.Parameter(torch.zeros((1)))
    
    def forward(self, x):
        '''
        performs a convolution and produces a feature map given an input
        
        input: tensor(batch_size, in_channels_num, input_size, input_size)
        output: tensor(batch_size, 1, output_size, output_size)
        '''
        
        # get dimensions
        batch_size, in_channels, input_size, input_size = x.shape
        # calculate output size
        output_size = input_size - self.size + 1
        
        assert self.in_channels == in_channels, 'input channel size mismatch with accepted num of channels'
        
        # inititlize output tensor
        # 1 for channel dimension
        output = torch.empty(batch_size, 1, output_size, output_size)
             
        ''' using for loops for readability and clarity, inefficiency doesn't matter here'''
        # vertical "slide" of convolution filter
        for i in range(output_size):
            
            # horizontal "slide" of convolution filter
            for j in range(output_size):
                
                # get patches using slicing to apply convolution on
                # i and j represent the top left pixel in the convolution patch
                if i == j == output_size-1:
                    input_patches = x[:, :, i:, j:]
                else:
                    input_patches = x[:, :, i:i+self.size, j:j+self.size]
                
                # apply convolution and save result in output
                # covolution: sum of hadmard product between input patch and kernel (dot product)
                output[:, 0, i, j] = torch.sum(input_patches * self.kernel) 
        
        # add bias
        # output: (batch_size, 1, output_size, output_size)
        return output + self.bias
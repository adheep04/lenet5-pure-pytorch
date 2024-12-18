import torch.nn as nn
import torch

from config import get_config


class LeNet_5(nn.Module):
    def __init__(self, config):  
        super().__init__()
        self.config = config
        # (b, 1, 32, 32) -> (b, 6, 28, 28)
        self.C1 = ConvolutionLayer(num_filters=6, filter_size=5, in_channels=1)
        # (b, 6, 28, 28) -> (b, 6, 14, 14)
        self.S2 = AvgPoolingLayer(num_channels=6)  
        # (b, 6, 14, 14) -> (b, 16, 10, 10)
        self.C3 = ConvolutionLayer(num_filters=16, filter_size=5, in_channels=6, connections=config['C3'])
        # (b, 16, 10, 10) -> (b, 16, 5, 5)
        self.S4 = AvgPoolingLayer(num_channels=6)  
        # (b, 16, 5, 5) -> (b, 120, 1, 1)
        self.C5 = ConvolutionLayer(num_filters=120, filter_size=5, in_channels=16,)
        # (b, 120, 1, 1) -> (b, 84)
        self.F6 = nn.Linear(in_features=120, out_features=84)
        


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

class RadialBasisFunctionLayer(nn.Module):
    def __init__(self, num_classes, size):
        self.num_classes = num_classes
        self.size = size
        self.weights = nn.Parameter(torch.tensor(self.num_classes, self.size))
        
    def forward(self, x, efficient):
        output = torch.empty(self.num_classes)
        
        if not efficient:
            for i in range(self.num_classes):
                output[i] = torch.sum((x - self.weights[i,:])**2)
        else:
            error_diff = torch.stack([x] * self.num_classes, dim=0) - self.weights
            output = torch.sum(error_diff**2, dim=1)
            
        assert torch.all(torch.isfinite(output)), f"output tensor not fully populated in RBF layer"
        
        return
         
        
         
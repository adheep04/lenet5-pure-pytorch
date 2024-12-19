import torch.nn as nn
import torch

from config import get_config


class LeNet_5(nn.Module):
    def __init__(self, config):  
        super().__init__()
        self.config = config
        self.A = 1.7159
        self.S = 2/3
        
        # define connection as a list of lists where the values of a list i correspond to 
        # the incoming channels of filter i's convolution in layer C3
        self.connections = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5],
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 1, 2, 5],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 1, 2, 3, 4, 5],
        ]
        
        self.pipeline = nn.Sequential(
            # (b, 1, 32, 32) -> (b, 6, 28, 28)
            ConvolutionLayer(num_filters=6, filter_size=5, in_channels=1,),
            TanhActivation(),
                            
            # (b, 6, 28, 28) -> (b, 6, 14, 14)
            AvgPoolingLayer(num_channels=6, config=config),
            TanhActivation(),
                    
            # (b, 6, 14, 14) -> (b, 16, 10, 10)
            ConvolutionLayer(num_filters=16, filter_size=5, in_channels=6, connections=self.connections),
            TanhActivation(),
                    
            # (b, 16, 10, 10) -> (b, 16, 5, 5)
            AvgPoolingLayer(num_channels=16, config=config),
            TanhActivation(),
                                                            
            # (b, 16, 5, 5) -> (b, 120, 1, 1)
            ConvolutionLayer(num_filters=120, filter_size=5, in_channels=16,),
            TanhActivation(),
                        
            # (b, 120, 1, 1) -> (b, 84)
            nn.Linear(in_features=120, out_features=84),
            TanhActivation(),
                                            
            # (b, 84) -> (b, 10,)
            RadialBasisFunctionLayer(weights=self.get_RBF_weights(), num_classes=10, size=84, config=config)
        )

        
        # initialize all layers individually
        self.C1 = ConvolutionLayer(num_filters=6, filter_size=5, in_channels=1)          # (b, 1, 32, 32) -> (b, 6, 28, 28)
        self.S2 = AvgPoolingLayer(num_channels=6, config=config)                         # (b, 6, 28, 28) -> (b, 6, 14, 14)
        self.C3 = ConvolutionLayer(                                                      # (b, 6, 14, 14) -> (b, 16, 10, 10)
            num_filters=16, filter_size=5, in_channels=6, connections=self.connections)
        self.S4 = AvgPoolingLayer(num_channels=16, config=config)                                        # (b, 16, 10, 10) -> (b, 16, 5, 5)
        self.C5 = ConvolutionLayer(num_filters=120, filter_size=5, in_channels=16)      # (b, 16, 5, 5) -> (b, 120, 1, 1)
        self.F6 = nn.Linear(in_features=120, out_features=84)                            # (b, 120, 1, 1) -> (b, 84)
        self.RBF = RadialBasisFunctionLayer(                                             # (b, 84) -> (b, 10,)
            weights=self.get_RBF_weights(), num_classes=10, size=84, config=config)
        self.tanh = TanhActivation()
    
    def forward(self, x):
        if not self.config['efficient_model']:
            x = self.tanh(self.C1(x))
            x = self.tanh(self.S2(x))
            x = self.tanh(self.C3(x))
            x = self.tanh(self.S4(x))
            x = self.tanh(self.C5(x))
            x = self.tanh(self.F6(x))
            return self.RBF(x)
            
        else:
            x = self.pipeline(x)
            return x
    
    def get_RBF_weights(self):
        # stylized 12x7 bitmaps for digits 0-9
        digits = [
            # Digit 0
            [
                "       ",
                "  ***  ",
                " *   * ",
                "*     *",
                "*     *",
                "*     *",
                "*     *",
                "*     *",
                " *   * ",
                " ** ** ",
                "  ***  ",
                "       ",
            ],
            # Digit 1
            [
                "   **  ",
                "  ***  ",
                " * **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                "   **  ",
                " ******",
                "       ",
            ],
            # Digit 2
            [
                " ***** ",
                " *   **",
                "     **",
                "     **",
                "    ** ",
                "   **  ",
                "  **   ",
                " **    ",
                " *     ",
                "**     ",
                "*******",
                "       ",
            ],
            # Digit 3
            [
                " ***** ",
                "**    *",
                "      *",
                "     * ",
                "    ** ",
                "  ***  ",
                "    ** ",
                "      *",
                "      *",
                "**   **",
                " ***** ",
                "       ",
            ],
            # Digit 4
            [
                "       ",
                "*     *",
                "**    *",
                "**    *",
                "*     *",
                "*******",
                "     **",
                "      *",
                "      *",
                "      *",
                "      *",
                "       ",
            ],
            # Digit 5
            [
                "       ",
                "*******",
                "*      ",
                "**     ",
                "**     ",
                "  **** ",
                "     **",
                "      *",
                "      *",
                "*    **",
                " ***** ",
                "       ",
            ],
            # Digit 6
            [
                " ***** ",
                "**     ",
                "*      ",
                "*      ",
                "****** ",
                "**   **",
                "*     *",
                "*     *",
                "*     *",
                "**    *",
                " ***** ",
                "       ",
            ],
            # Digit 7
            [
                "*******",
                "     **",
                "     **",
                "    ** ",
                "    *  ",
                "   **  ",
                "   *   ",
                "  **   ",
                "  **   ",
                "  *    ",
                "  *    ",
                "       ",
            ],
            # Digit 8
            [
                " ***** ",
                "**   **",
                "*     *",
                "**    *",
                " ***** ",
                "**   **",
                "*     *",
                "*     *",
                "*     *",
                "**   **",
                " ***** ",
                "       ",
            ],
            # Digit 9
            [
                " ***** ",
                "*     *",
                "*     *",
                "**    *",
                " ******",
                "      *",
                "      *",
                "      *",
                "      *",
                " *   **",
                " ***** ",
                "       ",
            ],
        ]
        
        bitmap = torch.empty(10, 12, 7)
        for d, digit in enumerate(digits):
            for j, row in enumerate(digit):
                for i, char in enumerate(row):
                    if char == "*":
                        bitmap[d, j, i] = 1.0
                    else:
                        bitmap[d, j, i] = -1.0
                        
        # validate no NaN's
        assert not torch.isnan(bitmap).any(), f"digit bitmap isn't being populated correctly"
        assert bitmap.shape == (10, 12, 7), f"incorrect digit bitmap dimensions"
                        
        # flatten to (10, 84)
        return bitmap.flatten(start_dim=1, end_dim=2)
        
class Convolution(nn.Module):
    '''
    defines a class for a square convolution with a stride of 1.
    **this class is not generalized for all convolutions, its for LeNet-5's purposes
    doens't use F.unfold
    
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
        self.kernel = nn.Parameter(data=torch.rand(size=(self.in_channels, self.size, self.size)))
        # bias scalar
        self.bias = nn.Parameter(torch.zeros((1)))
    
    def forward(self, x):
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
        
        # inititlize output tensor in the same device as the input
        output = torch.empty(batch_size, output_size, output_size, device=x.device)
             
        ''' using for loops for readability and clarity, inefficiency doesn't matter here'''
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
                output[:, i, j] = torch.sum(input_patches * self.kernel, dim=(1,2,3)) 
        
        # validate output tensor
        assert not torch.isnan(output).any(), f"output not fully populated at conv filter {self.in_channels}"
        
        # add bias
        # output: (batch_size, output_size, output_size)
        return output + self.bias
        
class ConvolutionLayer(nn.Module):
    '''
    defines a class for a single convolution layer in the LeNet-5 model
    
    args:
    - num_filters: int
    - filter_size: int
    - in_channels: int
    '''
    
    def __init__(self, in_channels, num_filters, filter_size, connections=None):
        super().__init__()
        
        # initialize useful fields
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
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
                convolution = filter(x)
            else:
                # (batch_size, 1, out_size, out_size)
                convolution = filter(self._get_input(i, x))

            # asign local convolution to corresponding output region
            output[:, i, :, :] = convolution
        
        # ensure output is fully populated by convolutions
        assert not torch.isnan(output).any(), f"output not fully populated at conv layer {self.num_filters}"
    
        return output.squeeze()
    
    def _get_input(self, i, x):
        
        # get list of inputs to filter i (e.g. [1, 2, 4, 5])
        input_list = self.connections[i]
        
        # use advanced indexing (passing in a list in slicing) to filter input
        return x[:, input_list, :, :]
    
class AvgPoolingLayer(nn.Module):
    '''
    an avg-pooling layer
    
    args:
    - num_channels: int
    '''
    def __init__(self, num_channels, config):
        super().__init__()
        self.num_channels = num_channels
        self.config = config
        self.weights = nn.Parameter(torch.ones(num_channels))
        self.biases = nn.Parameter(torch.zeros(num_channels))
    
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
        
        if not self.config['efficient']:
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
        weights = self.weights.reshape(1, self.num_channels, 1, 1)
        biases = self.biases.reshape(1, self.num_channels, 1, 1)

        # multiply weights by output and add bias
        # output: (b, c, s/2, s/2)
        return (weights * output) + biases

class RadialBasisFunctionLayer(nn.Module):
    '''
    computes the squared distance scalars between the input tensor and each
    corresponding weight vectors, which is a 7 x 12 bitmap of the digit of its class
    
    args:
    - num_classes: int -> 10
    - size: int -> 84
    - weights: tensor (num_classes, size) -> (10, 84)
    '''
    def __init__(self, weights, num_classes, size, config):
        super().__init__()
        self.num_classes = num_classes
        self.size = size
        self.config = config
        self.weights = nn.Parameter(weights, requires_grad=False) if weights is not None else nn.Parameter(torch.zeros(self.num_classes, self.size))
        
        
    def forward(self, x):
        '''
        args:
        - x: tensor: (batch_size, 84)
        
        output:
        - tensor: (batch_size, 10)
        '''
        
        # get batch size
        batch_size = x.shape[0]
        
        # creates (batch_size, 10,) tensor representing prediction values
        output = torch.empty(batch_size, self.num_classes, device=x.device)
        
        if not self.config['efficient']:
            # iterate through class numbers and calculate prediction
            for i in range(self.num_classes):
                diff = x - self.weights[i,:]
                # prediction is the difference between input vector and weight 
                # vector for a given class representing a 7x12 of the digit 
                output[:, i] = torch.sum(diff**2, dim=1)
        else:
            # (batch_size, 84) -> (batch_size, 1, 84)
            x = x.unsqueeze(1)
            # (10, 84) -> (1, 10, 84)
            w = self.weights.unsqueeze(0)
            # (batch_size, 1, 84) - (1, 10, 84) -> (batch_size, 10, 84)
            error_diff = x - w
            # (batch_size, 10, 84) -> (batch_size, 10,)
            output = torch.sum(error_diff**2, dim=2)
            
        # validate output tensor
        assert not torch.isnan(output).any(), f"output tensor not fully populated in RBF layer"
        
        # return (b, 10)
        return output

class TanhActivation(nn.Module):
    # define static class attribute
    tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * TanhActivation.tanh((2 / 3) * x)
    
class LossFunction(nn.Module):
    def forward(self, preds, labels, batch=True):
        first_terms = preds[torch.arange(preds.size(0)), labels]
        
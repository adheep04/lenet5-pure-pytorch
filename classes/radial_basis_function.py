from torch import nn
import torch

class RadialBasisFunctionLayer(nn.Module):
    '''
    computes the squared distance scalars between the input tensor and each
    corresponding weight vectors, which is a 7 x 12 bitmap of the digit of its class
    
    args:
    - num_classes: int -> 10
    - size: int -> 84
    - weights: tensor (num_classes, size) -> (10, 84)
    '''
    def __init__(self, weights, num_classes, size, efficient):
        super().__init__()
        self.num_classes = num_classes
        self.size = size
        self.efficient = efficient
        self.bitmap_weight = nn.Parameter(weights, requires_grad=False) if weights is not None else nn.Parameter(torch.zeros(self.num_classes, self.size), requires_grad=False)
        
        
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
        
        if not self.efficient:
            # iterate through class numbers and calculate prediction
            for i in range(self.num_classes):
                diff = x - self.bitmap_weight[i,:]
                # prediction is the difference between input vector and weight 
                # vector for a given class representing a 7x12 of the digit 
                output[:, i] = torch.sum(diff**2, dim=1)
        else:
            # (batch_size, 84) -> (batch_size, 1, 84)
            x = x.unsqueeze(1)
            # (10, 84) -> (1, 10, 84)
            w = self.bitmap_weight.unsqueeze(0)
            # (batch_size, 1, 84) - (1, 10, 84) -> (batch_size, 10, 84)
            error_diff = x - w
            # (batch_size, 10, 84) -> (batch_size, 10,)
            output = torch.sum(error_diff**2, dim=2)
            
        # validate output tensor
        assert not torch.isnan(output).any(), f"output tensor not fully populated in RBF layer"
        
        # return (b, 10)
        return output

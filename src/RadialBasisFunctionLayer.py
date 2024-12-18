import torch
from torch import nn

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
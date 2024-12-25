from torch import nn
import torch

class MaxAPosteriroiLoss(nn.Module):
    def forward(self, preds, labels, j):
        
        # turns scalar into tensor
        j = torch.tensor(j)
        
        # get batch size
        batch_size = preds.shape[0]
        
        # uses pytorch's advanced indexing to get the correct prediction
        # values for the sample/s in the batch: (batch_size, 1)
        y_true = preds[torch.arange(batch_size), labels].reshape(-1)
        
        # scalar term that controls ratio of MAP portion of loss
        exp_term = torch.exp(-j)
        
        # raises e by all 10 predictions for all 16 samples and sum 
        # all prediction values for each sample in the batch:
        # (batch_size,1)
        sum_terms = torch.sum(torch.exp(-preds), dim=1)
        
        # return mean of all predictions for sample/s 
        return torch.mean(y_true + torch.log(exp_term + sum_terms))
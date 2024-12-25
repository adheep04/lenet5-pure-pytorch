from torch import nn

class TanhActivation(nn.Module):
    '''
    "squashing function" authors of LeNet paper use for each layer. 
    involves scaling layer output by 2/3 (S), applying tanh, and multiplying by
    1.7159 (A)
    '''
    # define static class attribute
    tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * TanhActivation.tanh((2 / 3) * x)
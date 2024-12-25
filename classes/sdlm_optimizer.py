from torch.optim import Optimizer
import torch

# implementation of Stochastic Diagonal Levenberg-Marquard 
# optimizer as specified in appendix c
class SDLMOptimizer(Optimizer):
    def __init__(self, params, lr, safety, mu):
        # put hyperparams in a dict and initialize it using superclass
        hyperparams = dict(lr=lr, safety=safety, mu=mu)
        # initializes optimizer.state, optimizer.param_groups, and hyperparameters
        super().__init__(params, hyperparams)

        # initialize hessian attribute for each parameter in state dict
        for param in self.param_groups[0]['params']:
            self.state[param]['hessian'] = torch.zeros_like(param)
                
    def update_lr(self,lr):
        self.param_groups[0]['lr'] = lr
    
    def step(self):
        # get hyperparameters
        lr = self.param_groups[0]['lr']
        safety = self.param_groups[0]['safety']
        mu = self.param_groups[0]['mu']
        
        # iterate through every parameter tensor
        for param in self.param_groups[0]['params']:
            
            # get current hessian estimate
            old_hessian = self.state[param]['hessian']
            
            # update hessian estimate using the square of gradient according to appendix c
            
            # square gradient of parameter
            gradient_squared = param.grad.data ** 2
            
            # dampening value (0.99) * old_hessian value + 0.01 * grad*grad
            # grad*grad is an approximation/replacement of the second derivative
            new_hessian = mu * old_hessian + (1-mu) * gradient_squared
            self.state[param]['hessian'] = new_hessian
        
            # safety adds noise to the hessian signal (controls how much or how little)
            # the larger the second derivative, the smaller the step size
            step_size = lr / (safety + new_hessian)
            
            # calculate update tensor
            update = -step_size * param.grad.data
            
            # update weight value -> e_k * w_k
            param.data.add_(update)
        return update
        
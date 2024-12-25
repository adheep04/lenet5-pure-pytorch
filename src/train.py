from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch

from model import LeNet_5, SDLMOptimizer, MaxAPosteriroiLoss
from config import get_config

# Define transformation
def get_transform():
    return transforms.Compose([
    
        # add padding since input from MNIST is 28x28 instead of 
        # model expected 32x32
        transforms.Pad(padding=2),
        
        # converts image to tensor normalized to range [0, 1]
        transforms.ToTensor(),
        
        # apply normalization used in paper [-0.1, 1.175]
        transforms.Lambda(lambda x: x * (1.175 + 0.1) - 0.1)
])

# initialize the model weights according to Appendix C in paper
def initialize_model_weights(model, device): 
    # fan-in value for all neurons in the given layer.
    # C3's (4) neurons have different fan-ins
    fan_in_map = {
        'C1' : 25,
        'S2' : 4,
        'C3' : None,
        'S4' : 4,
        'C5' : 400,
        'F6' : 120        
    }

    # init weight value with random num from (-2.4/fan_in, 2.4/fan_in) as in paper
    def initialize_weight(fan_in, shape):
        return torch.empty(size=shape).uniform_(-2.4/fan_in, 2.4/fan_in)
    
    # list where the ith element represents the number of incoming channels to filter in in layer C3
    in_channels_nums = [len(in_channels) for in_channels in model.connections]

    # iterate through parameters
    for name, weight in model.named_parameters():
        # skip bias parameters as they are already init to 0
        if ('bias' in name) or ('RBF' in name):
            continue
        
        # get weight tensor shape, fan_in number, and layer number
        shape = weight.data.shape
        layer = name[9:11]
        fan_in = fan_in_map[layer]
        
        # neurons in C3 have different fan_in values
        if 'C3' in name:
            # extract filter number from name string
            try:
                # if number is 2 digits 
                filter_num = int(name[20:22])
            except:
                # if number is 1 digit
                filter_num = int(name[20:21])
            # fan_in = kernel height * kernel width * input channel number
            fan_in = 25 * in_channels_nums[filter_num]
        
        # initialize weight
        weight.data = initialize_weight(fan_in=fan_in, shape=shape).to(device)
        
def print_stats(training_stats):
    print()
    print('_______CHECKPOINTS________')
    for step in training_stats:
        print('_________________________________________')
        for stat in step:
            print(stat, ': ',step[stat])
        print()
    print()
    print()
    print()
    print('______________FINAL STATS________________')
    correct = sum(stat['correct'] for stat in training_stats)
    total = len(training_stats)
    print(f'Final Accuracy: {correct/total*100:.2f}%')
    #print(f'Average Loss: {sum(stat["avg_loss"] for stat in training_stats)/total:.4f}')
    print('_________________________________________')
    
def run_validation(test_data, model, loss_fn, num_batches, j, device=torch.device('cuda')):
    
    # if data is a dataloader
    is_dataloader = isinstance(test_data, torch.utils.data.DataLoader)
    
    # total samples in test run
    if is_dataloader:
        total_samples = num_batches * test_data.batch_size
    # if data is a list
    else:
        total_samples = len(test_data)
        
    # initialize accumulative variables
    total_correct = 0
    loss = 0
    
    # initialize per-digit f1 metrics
    per_digit = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(10)} 
    
    # put model in right device
    model = model.to(device)
    
    for step, (data, labels) in enumerate(test_data):
        if step == num_batches:
            break
        
        # send data to device
        data = data.to(device)
        
        # add extra dimension if label has 1 dimension
        if not is_dataloader:
            labels = labels.unsqueeze(0).to(device)
        
        # (batch_size, 10)
        preds = model(data)
        
        # (batch_size, 10) -> (batch_size, 1)
        # gets the model prediction
        readable_pred = torch.argmin(preds, dim=1)
        
        # gets the number of correct answers
        correct = sum(1 for i, pred in enumerate(preds) if torch.argmin(pred) == labels[i])
        
        # update total
        total_correct += correct
        
        # calculate loss
        loss_batch = loss_fn(preds=preds, labels=labels, j=j)
        
        # update loss
        loss += loss_batch
        
        # Update f1 metrics
        pred_digits = readable_pred.detach().cpu().numpy()
        true_digits = labels.detach().cpu().numpy()
        
        for true, pred in zip(true_digits, pred_digits):
            # True positive for the correct digit
            if true == pred:
                per_digit[true]["tp"] += 1
            else:
                # False negative for the true digit
                per_digit[true]["fn"] += 1
                # False positive for the predicted digit
                per_digit[pred]["fp"] += 1
            
        
        print()
        print(f'__________batch test__________')
        if is_dataloader:
            print(f'accuracy: {correct/test_data.batch_size*100}%\n')
        else:
            print(f'accuracy: {True if correct>0 else False}% \n')
        print(f'PRED:')
        print(readable_pred.detach().cpu().numpy().tolist())
        print()
        print(f'LABEL:')
        print(labels.detach().cpu().numpy().tolist())
        print()
        
        
    # Calculate F1 scores
    f1_scores = {}
    macro_f1 = 0
    
    for digit in per_digit:
        tp = per_digit[digit]["tp"]
        fp = per_digit[digit]["fp"]
        fn = per_digit[digit]["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[digit] = f1
        macro_f1 += f1
    
    # average stats
    macro_f1 /= 10  
    accuracy = total_correct/total_samples*100
    avg_loss = loss/num_batches
    
    print()
    print('_____validation results______')
    print(f'accuracy: {accuracy:.2f}%')
    print(f'average loss: {avg_loss}\n')
    print(f'per_digit_f1: {f1_scores}\n')
    print(f'macro_f1: {macro_f1} \n')
    return {
        'accuracy' : accuracy,
        'average_loss' : avg_loss,
        'per_digit_f1': f1_scores,
        'macro_f1': macro_f1
    }
        
    
def train_model(run_file=None):
    
    if run_file is not None:
        resuming_training = True 
        last_epoch = run_file['run_stats'][-1]['epoch']
        m_state_dict = run_file['model_state_dict']
        o_state_dict = run_file['optimizer_state_dict']
        run_stats = run_file['run_stats']
    else:
        resuming_training = False
    
    # get config
    config = get_config()
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # initialize model
    model = LeNet_5(config=config).to(device)
    
    # load from state dict
    if resuming_training:
        print('loading previous model state...')
        model.load_state_dict(m_state_dict)
    
    # initialize model weights for training if no state_dict is supplied
    if not resuming_training:
        print('initializing weights for new model...')
        initialize_model_weights(model,device=device)
    
    # load train and test data
    transform = get_transform()
    train_data = datasets.MNIST(root=config['train_path'], train=True, transform=transform, download=True,)
    test_data = datasets.MNIST(root=config['test_path'], train=False, transform=transform, download=True,)
    
    # initialize train dataloader
    train_dataloader = DataLoader(
        dataset=train_data, 
        batch_size=config['batch_size'],
        shuffle=True,
        )
    
    # initialize test dataloader
    test_dataloader = DataLoader(
        dataset=test_data, 
        batch_size=config['test_batch_size'],
        shuffle=True,
        )
    
    # 2nd derivative update dataloader
    # yields 500 random samples from training set
    hessian_dataloader = DataLoader(
        dataset=train_data, 
        batch_size=500,
        shuffle=True,
        )
    
    # get custom defined loss function
    loss_fn = MaxAPosteriroiLoss()
    
    # get all params that require gradients
    params = [param for param in model.parameters() if param.requires_grad]
    
    # initialize optimizer
    optimizer = SDLMOptimizer(params=params, lr=config['lr'][0], safety=config['damping'], mu=config['mu'])
    
    # reload optimizer state
    if resuming_training:
        print('reloading previous optimizer state...')
        optimizer.load_state_dict(o_state_dict)
    
    # initialize training stats
    training_stats = []
    
    # initialize validation per epoch stats
    validation_stats = [{'accuracy' : 96.22, 'average_loss' : 4.4089}]
    
    # reload stats list
    if resuming_training:
        training_stats = run_stats
    
    try:
        if resuming_training:
            print(f'resuming training on epoch {last_epoch}...')
        # start training loop
        for epoch in range(last_epoch+1 if resuming_training else 0, config['num_epochs']):
            
            # update learning rate
            optimizer.update_lr(lr=config['lr'][epoch])
            
            # re-estimate second derivative of each weight with 
            # respect to the loss using 500 random samples
            print('estimating new hessian values for epoch...')
            for step, (data, label) in enumerate(hessian_dataloader):
                if step > 0:
                    break
                
                # move data to CUDA
                data = data.to(device)
                label = label.to(device)
                
                # forward pass through model
                pred = model(data)
                
                # calculate loss
                loss = loss_fn(pred, label, j=config['j'])

            # iterate through every parameter
            for group in optimizer.param_groups:
                for param in group['params']:
                    # calculate first derivative of weight with respect to loss
                    p_grad = torch.autograd.grad(loss, [param], create_graph=True)[0]
                    # calculate second derivative of weight with respect to loss
                    p_second_der = torch.autograd.grad(p_grad.sum(), [param], retain_graph=True)[0]
                    # validate dimensions 
                    assert optimizer.state[param]['hessian'].shape == p_second_der.shape, 'dimensional mismatch in second derivative for hessian calculation'
                    # update hessian value for the weight
                    optimizer.state[param]['hessian'] = p_second_der
                    
                    
            # start main training loop
            print()
            if run_file is None:
                print(f'starting epoch {epoch + 1}...')
            
            # variable to log average loss every 100 steps
            loss_mean_temp = 0
        
            for step, (data, label) in enumerate(train_dataloader):
                # move data to CUDA
                data = data.to(device)
                label = label.to(device)
                
                # re-initialize gradients
                optimizer.zero_grad()
                
                # forward pass to get prediction
                pred = model(data)
                
                # get the digit prediction for each batch
                readable_pred = torch.argmin(pred, dim=1)
                
                # calculate max a posteriori loss
                loss = loss_fn(preds=pred, labels=label, j=config['j'])
                
                # calculate gradients with respect to loss
                loss.backward()
                
                # update weights
                optimizer.step()
                print('STEP: ', step+1)
                print(f'LOSS: {loss.item()}')
                print(f'MODEL PRED: {torch.tensor(pred[0])}')
                print(f'MODEL PRED: {readable_pred[0].item()}') 
                print(f'TRUE DIGIT: {label[0].item()}')         
                print('______________________________________')
                
                # update loss temp for stats
                loss_mean_temp += loss.item()
                
                # add training stats
                if step%100 == 0:
                    training_stats.append({
                        'epoch' : epoch,
                        'step': step,
                        'avg_loss': loss_mean_temp/100 if step > 0 else loss_mean_temp,
                        'step_loss': loss.item(),
                        'step output' : {torch.tensor(pred[0])},
                        'predictions': readable_pred.tolist(),  
                        'true_labels': label.tolist(),         
                        'correct': (readable_pred == label).float().mean().item(), 
                        'pred_confidence': torch.max(torch.softmax(-pred, dim=1)).item(),
                    })
                    
                    # reset mean loss for next set stats
                    loss_mean_temp = 0
                    
            validation_stats.append(run_validation(
                test_data=test_dataloader,
                model=model,
                loss_fn=MaxAPosteriroiLoss(),
                num_batches=100,
                j=1
            ))
    except:
        pass
    # save file
    torch.save({
            'run_stats' : training_stats,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'validation_stats' : validation_stats
        }, f'./checkpoints/run_j_higher.pt')
    print_stats(training_stats)
    return

# try:
#     dict = torch.load('./checkpoints/model_epoch_1.pt')
#     train_model(dict)
# except:
# train_model()











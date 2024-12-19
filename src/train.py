from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch

from model import LeNet_5
from config import get_config

# Define transformation
def get_transform():
    return transforms.Compose([
    
        # add padding since input from MNIST is 28x28 instead of 
        # model expected 32x32
        transforms.Pad(padding=2),
        
        # converts image to tensor normalized to range [0, 1]
        transforms.ToTensor(),
        
        # normalization used in paper [-0.1, 1.175]
        transforms.Lambda(lambda x: x * (1.175 + 0.1) - 0.1)
])
    

def train_model():
    # get config
    config = get_config
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # initialize model
    model = LeNet_5(config=config).to(device)
    
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
        batch_size=1,
        shuffle=True,
        )
    
    for epoch in range(config['num_epochs']):
        print(f'starting epoch {epoch}')
        for step, (data, label) in enumerate(train_dataloader):
            pass

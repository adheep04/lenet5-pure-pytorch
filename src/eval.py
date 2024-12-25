import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from train import get_transform, run_validation
from config import get_config
from model import LeNet_5, MaxAPosteriroiLoss

def print_stats(training_stats):
    print()
    print('_______CHECKPOINTS________')
    for step in training_stats:
        print('_________________________________________')
        for stat in step:
            if stat == 'avg_loss':
                print(stat, ': ',step[stat])
        print()
    print()
    print()
    print()
    print('______________FINAL STATS________________')
    correct = sum(stat['correct'] for stat in training_stats)
    total = len(training_stats)
    print(f'Final Accuracy: {correct/total*100:.2f}%')
    print(f'Average Loss: {sum(stat["avg_loss"] for stat in training_stats)/total:.4f}')
    print('_________________________________________')
    
# Define transformation
def transform_img():
    
    transform = transforms.Compose([
        
        # converts image to tensor normalized to range [0, 1]
        transforms.ToTensor(),
        
        # convert to grayscale
        transforms.Grayscale(num_output_channels=1),
        
        # apply normalization used in paper [-0.1, 1.175]
        transforms.Lambda(lambda x: x * (1.175 + 0.1) - 0.1)
])
    return transform

dict = torch.load('./checkpoints/model_epoch_1.pt')

config = get_config()

# load train and test data
transform = get_transform()
test_data = datasets.MNIST(root=config['test_path'], train=False, transform=transform, download=True,)

# initialize test dataloader
test_dataloader = DataLoader(
    dataset=test_data, 
    batch_size=config['test_batch_size'],
    shuffle=True,
    )

# initialize model
model = LeNet_5(config=config).to(torch.device('cuda'))

print('running validation without training...')
run_validation(
    test_data=test_dataloader,
    model=model,
    loss_fn=MaxAPosteriroiLoss(),
    num_batches=config['test_num_batches'],
    j=1
)

print('loading previous model state...')
model.load_state_dict(dict['model_state_dict'])
run_validation(
    test_data=test_dataloader,
    model=model,
    loss_fn=MaxAPosteriroiLoss(),
    num_batches=config['test_num_batches'],
    j=1
)

dataset = [(torch.tensor(transform_img()(Image.open(img))).unsqueeze(0), torch.tensor(int(img.name[0]))) for img in Path("./data/custom_digits").iterdir()]


losses = [dict['run_stats'][i]['avg_loss'] for i in range(len(dict['run_stats']))]
step_losses = [dict['run_stats'][i]['step_loss'] for i in range(len(dict['run_stats']))]

plt.plot(range(len(losses)), losses, alpha=0.7, label='avg loss for last 100 steps')
plt.title('average training loss over time')
plt.xlabel('training steps (per hundred)')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(range(len(step_losses)), step_losses, alpha=0.7, label='raw loss')
plt.title('step loss over time')
plt.xlabel('Training Steps (per hundred)')
plt.ylabel('Loss')
plt.legend()
plt.show()


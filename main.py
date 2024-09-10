import torch
from torch.utils.data import DataLoader
from data.dataset import DenoisingDataset
from models.generator import UNetGenerator
from models.discriminator import Discriminator
from train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
noisy_dir = './data/noisy'
clean_dir = './data/clean'
batch_size = 16
epochs = 50
lr = 0.0002

# Data preparation
dataset = DenoisingDataset(noisy_dir, clean_dir, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
generator = UNetGenerator()
discriminator = Discriminator()

# Training
train(generator, discriminator, dataloader, epochs, device)

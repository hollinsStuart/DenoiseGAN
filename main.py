import torch
from torch.utils.data import DataLoader
from train import train
from models.generator import UNetGenerator
from models.discriminator import Discriminator
from data.dataset import DenoisingDataset
import torchvision.transforms as transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
epochs = 100
batch_size = 64
num_classes = 2  # Set the number of classes

# Define image transformations (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range
])

# Dataset and Dataloader
noisy_dir = 'path/to/noisy/images'
clean_dir = 'path/to/clean/images'
label_file = 'path/to/labels.txt'  # File containing labels

# Initialize the dataset with the labels file
dataset = DenoisingDataset(noisy_dir=noisy_dir, clean_dir=clean_dir, label_file=label_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
generator = UNetGenerator(num_classes=num_classes)
discriminator = Discriminator(num_classes=num_classes)

# Training
train(generator, discriminator, dataloader, epochs, device, num_classes)

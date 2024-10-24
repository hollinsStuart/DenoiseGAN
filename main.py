import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.dataset import DenoisingDataset
from models.discriminator import Discriminator
from models.generator import UNetGenerator
from train import train

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
epochs = 5
batch_size = 64
num_classes = 2  # Set the number of classes

# Define image transformations (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] range
])

# Dataset and Dataloader
noisy_dir = './data/random_greyscale_images/class1'
clean_dir = './data/random_greyscale_images/class2'
label_file = './data/random_greyscale_images/fake_labels.txt'  # File containing labels

# Initialize the dataset with the labels file
dataset = DenoisingDataset(noisy_dir=noisy_dir, clean_dir=clean_dir, label_file=label_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
generator = UNetGenerator(num_classes=num_classes)
discriminator = Discriminator(num_classes=num_classes)

# Training
g_loss, d_loss = train(generator, discriminator, dataloader, epochs, device, num_classes)


def plot_loss(g_loss_history, d_loss_history):
    plt.figure(figsize=(10, 5))

    # Plot generator loss
    plt.plot(g_loss_history, label='Generator Loss', color='blue')

    # Plot discriminator loss
    plt.plot(d_loss_history, label='Discriminator Loss', color='red')

    # Adding titles and labels
    plt.title('Generator and Discriminator Loss During Training')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()


# Example usage (after training):
plot_loss(g_loss, d_loss)

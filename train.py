import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

# Create directories to store results
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('generated_images', exist_ok=True)


def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def train(generator, discriminator, dataloader, epochs, device, num_classes):
    criterion = nn.MSELoss()
    adv_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    generator.to(device)
    discriminator.to(device)

    g_loss_history = []
    d_loss_history = []

    for epoch in range(epochs):
        for i, (noisy_images, clean_images, labels) in enumerate(dataloader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)
            labels = F.one_hot(labels, num_classes).float().to(device)  # Convert labels to one-hot

            # Training Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(clean_images.size(0), 1).to(device)
            fake_labels = torch.zeros(clean_images.size(0), 1).to(device)

            # Real images + real labels
            outputs = discriminator(clean_images, labels)
            d_loss_real = adv_loss(outputs, real_labels)

            # Fake images + real labels
            fake_images = generator(noisy_images, labels)
            outputs = discriminator(fake_images.detach(), labels)
            d_loss_fake = adv_loss(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Training Generator
            optimizer_G.zero_grad()
            fake_images = generator(noisy_images, labels)
            outputs = discriminator(fake_images, labels)
            g_loss = adv_loss(outputs, real_labels) + criterion(fake_images, clean_images)

            g_loss.backward()
            optimizer_G.step()

            # Save loss values for tracking
            g_loss_history.append(g_loss.item())
            d_loss_history.append(d_loss.item())

            # Print losses
            print(
                f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

            # Save generated images every 100 steps
            if (i + 1) % 100 == 0:
                save_image(fake_images.data[:25], f'generated_images/fake_image_epoch_{epoch + 1}_step_{i + 1}.png',
                           nrow=5, normalize=True)

        # Save model checkpoints every epoch
        save_checkpoint(generator, optimizer_G, epoch + 1, f'checkpoints/generator_epoch_{epoch + 1}.pth')
        save_checkpoint(discriminator, optimizer_D, epoch + 1, f'checkpoints/discriminator_epoch_{epoch + 1}.pth')

    return g_loss_history, d_loss_history

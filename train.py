import torch
import torch.optim as optim
from models.generator import UNetGenerator
from models.discriminator import Discriminator
from utils.loss import FocalLoss
import torch.nn as nn
import torch.nn.functional as F


def train(generator, discriminator, dataloader, epochs, device, num_classes):
    criterion = nn.MSELoss()
    adv_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    generator.to(device)
    discriminator.to(device)

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

            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

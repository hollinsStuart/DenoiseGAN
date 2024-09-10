import torch
import torch.optim as optim
from models.generator import UNetGenerator
from models.discriminator import Discriminator
from utils.loss import FocalLoss

def train(generator, discriminator, dataloader, epochs, device):
    criterion = nn.MSELoss()
    adv_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        for i, (noisy_images, clean_images) in enumerate(dataloader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Training Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(clean_images.size(0), 1).to(device)
            fake_labels = torch.zeros(clean_images.size(0), 1).to(device)

            outputs = discriminator(clean_images)
            d_loss_real = adv_loss(outputs, real_labels)

            fake_images = generator(noisy_images)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = adv_loss(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Training Generator
            optimizer_G.zero_grad()
            fake_images = generator(noisy_images)
            outputs = discriminator(fake_images)
            g_loss = adv_loss(outputs, real_labels) + criterion(fake_images, clean_images)

            g_loss.backward()
            optimizer_G.step()

            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')


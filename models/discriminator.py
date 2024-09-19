import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        # Modify input to accept image channels + labels concatenated
        input_channels = 1 + num_classes  # 1 channel for grayscale image + num_classes for one-hot labels

        self.model = nn.Sequential(
            self.block(input_channels, 64, batch_norm=False),
            self.block(64, 128),
            self.block(128, 256),
            self.block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def block(self, in_channels, out_channels, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, img, labels):
        # Concatenate the image with the labels
        labels = labels.view(labels.size(0), -1, 1, 1)  # Reshape labels to match image dimensions
        labels = labels.expand(labels.size(0), labels.size(1), img.size(2), img.size(3))  # Expand to match img
        img_with_labels = torch.cat([img, labels], dim=1)
        return self.model(img_with_labels)

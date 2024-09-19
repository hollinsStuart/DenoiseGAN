import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    def __init__(self, num_classes):
        super(UNetGenerator, self).__init__()

        # Concatenate noise and class labels at the input
        input_dim = 100 + num_classes  # noise vector length (100) + number of classes

        # Modify the first layer to accept the concatenated input
        self.down1 = self.down_block(input_dim, 64)  # Adjust input dimension
        self.down2 = self.down_block(64, 128)
        self.down3 = self.down_block(128, 256)
        self.down4 = self.down_block(256, 512)

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )

        # Upsampling layers (decoder with skip connections)
        self.up4 = self.up_block(1024, 512)
        self.up3 = self.up_block(512 + 512, 256)  # Skip connection
        self.up2 = self.up_block(256 + 256, 128)
        self.up1 = self.up_block(128 + 128, 64)

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z, labels):
        # Concatenate noise vector (z) and one-hot encoded labels
        z = torch.cat([z, labels], dim=1)
        z = z.view(z.size(0), -1, 1, 1)  # Reshape to match Conv2D input
        return self.final(self.up1(self.down1(z)))  # Adjust based on architecture

    # Rest of the UNetGenerator remains unchanged

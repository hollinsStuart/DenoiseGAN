import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        # Downsampling layers (encoder)
        self.down1 = self.down_block(1, 64)
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
        self.up3 = self.up_block(512 + 512, 256)  # Skip connection: concatenates with corresponding encoder output
        self.up2 = self.up_block(256 + 256, 128)
        self.up1 = self.up_block(128 + 128, 64)

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Encoder part
        d1 = self.down1(x)  # First down-sampling
        d2 = self.down2(d1)  # Second down-sampling
        d3 = self.down3(d2)  # Third down-sampling
        d4 = self.down4(d3)  # Fourth down-sampling

        # Bottleneck
        b = self.bottleneck(d4)

        # Decoder part with skip connections
        u4 = self.up4(b)
        u3 = self.up3(torch.cat([u4, d4], 1))  # Skip connection with d4
        u2 = self.up2(torch.cat([u3, d3], 1))  # Skip connection with d3
        u1 = self.up1(torch.cat([u2, d2], 1))  # Skip connection with d2

        # Final output layer (skip connection with d1)
        output = self.final(torch.cat([u1, d1], 1))

        return output

import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        # Downsampling layers (encoder)
        self.encoder = nn.Sequential(
            self.down_block(1, 64),
            self.down_block(64, 128),
            self.down_block(128, 256),
            self.down_block(256, 512),
        )
        # Upsampling layers (decoder)
        self.decoder = nn.Sequential(
            self.up_block(512, 256),
            self.up_block(256, 128),
            self.up_block(128, 64),
            self.up_block(64, 1, output_layer=True),
        )

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def up_block(self, in_channels, out_channels, output_layer=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        if output_layer:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

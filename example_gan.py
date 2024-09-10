import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()
        self.encoder = nn
        
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss

class AdversarialLoss(nn.Module):
    """Adversarial loss for the cGAN"""
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, outputs, labels):
        return self.bce(outputs, labels)

# Combine the Focal Loss with the adversarial loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.adversarial_loss = AdversarialLoss()

    def forward(self, real_output, fake_output, real_labels, fake_labels):
        real_loss = self.adversarial_loss(real_output, real_labels)
        fake_loss = self.adversarial_loss(fake_output, fake_labels)
        return (real_loss + fake_loss) / 2

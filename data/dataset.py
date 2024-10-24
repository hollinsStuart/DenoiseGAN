import os
import numpy as np
from torch.utils.data import Dataset
import torch


class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, label_file, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        self.transform = transform

        # Load labels from a file
        with open(label_file, 'r') as f:
            self.labels = [int(label.strip()) for label in f.readlines()]

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        # Load .npy files as numpy arrays
        noisy_image = np.load(os.path.join(self.noisy_dir, self.noisy_images[idx]))
        clean_image = np.load(os.path.join(self.clean_dir, self.clean_images[idx]))

        # Convert numpy arrays to torch tensors
        noisy_image = torch.from_numpy(noisy_image).unsqueeze(0).float()  # Add channel dimension and convert to float
        clean_image = torch.from_numpy(clean_image).unsqueeze(0).float()  # Add channel dimension and convert to float

        # If you have fewer labels than images, cycle through the labels
        label = self.labels[idx % len(self.labels)]  # Cycle through labels if not enough for all images

        # Apply transform only if it's set and if the input is not already a tensor
        if self.transform:
            if isinstance(noisy_image, torch.Tensor) and isinstance(clean_image, torch.Tensor):
                # Skip transformation if it's already a tensor
                noisy_image = noisy_image  # Optionally apply tensor-compatible transforms here
                clean_image = clean_image  # Optionally apply tensor-compatible transforms here
            else:
                noisy_image = self.transform(noisy_image)
                clean_image = self.transform(clean_image)

        return noisy_image, clean_image, label  # Return the label as well

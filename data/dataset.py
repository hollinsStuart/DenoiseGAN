import os
import numpy as np
from torch.utils.data import Dataset
import torch

import pydicom

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


class XRayDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, label_file, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        self.transform = transform

        # Load labels from a file
        with open(label_file, 'r') as f:
            self.labels = [int(label.strip()) for label in f.readlines()]

        # Filter out files without pixel data
        self._filter_files()

    def _filter_files(self):
        """Remove files without pixel data from the lists."""
        valid_noisy_images = []
        valid_clean_images = []
        valid_labels = []

        for idx, (noisy_file, clean_file) in enumerate(zip(self.noisy_images, self.clean_images)):
            noisy_path = os.path.join(self.noisy_dir, noisy_file)
            clean_path = os.path.join(self.clean_dir, clean_file)

            try:
                noisy_dicom = pydicom.dcmread(noisy_path)
                clean_dicom = pydicom.dcmread(clean_path)

                if hasattr(noisy_dicom, 'PixelData') and hasattr(clean_dicom, 'PixelData'):
                    valid_noisy_images.append(noisy_file)
                    valid_clean_images.append(clean_file)
                    valid_labels.append(self.labels[idx % len(self.labels)])  # Cycle through labels if needed

            except Exception as e:
                print(f"Skipping files at index {idx} due to error: {e}")

        # Update file lists with valid files only
        self.noisy_images = valid_noisy_images
        self.clean_images = valid_clean_images
        self.labels = valid_labels

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_image_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_image_path = os.path.join(self.clean_dir, self.clean_images[idx])

        noisy_dicom = pydicom.dcmread(noisy_image_path)
        clean_dicom = pydicom.dcmread(clean_image_path)

        # Convert pixel data to numpy arrays
        noisy_image = noisy_dicom.pixel_array.astype(np.float32)
        clean_image = clean_dicom.pixel_array.astype(np.float32)

        # Apply transform if available and if not already a tensor
        if self.transform:
            noisy_image = self.transform(noisy_image) if isinstance(noisy_image, np.ndarray) else noisy_image
            clean_image = self.transform(clean_image) if isinstance(clean_image, np.ndarray) else clean_image

        # Convert to torch tensors only if still numpy arrays
        if isinstance(noisy_image, np.ndarray):
            noisy_image = torch.from_numpy(noisy_image).unsqueeze(0).float()
        if isinstance(clean_image, np.ndarray):
            clean_image = torch.from_numpy(clean_image).unsqueeze(0).float()

        # Get the label
        label = self.labels[idx]

        return noisy_image, clean_image, label
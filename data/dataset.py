import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        self.transform = transform

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_image = Image.open(os.path.join(self.noisy_dir, self.noisy_images[idx])).convert('L')
        clean_image = Image.open(os.path.join(self.clean_dir, self.clean_images[idx])).convert('L')

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

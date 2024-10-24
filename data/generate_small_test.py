import numpy as np
import os

# Directory to save the images
output_dir = 'class2'
os.makedirs(output_dir, exist_ok=True)

# Number of images
num_images = 50
# Image dimensions
height, width = 128, 128

for i in range(num_images):
    # Generate a random 128x128 8-bit greyscale image (values between 0 and 255)
    image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    # Save the image as a .npy file
    npy_filename = os.path.join(output_dir, f'image_{i+1}.npy')
    np.save(npy_filename, image)

print(f"Generated and saved {num_images} random greyscale images in {output_dir}")

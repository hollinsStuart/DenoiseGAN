import os
import random

def generate_random_labels(num_images, num_classes, output_file):
    """
    Generates random labels for a dataset and saves them in a text file.

    Args:
        num_images (int): The number of images in the dataset.
        num_classes (int): The number of possible classes for the labels.
        output_file (str): The file to save the generated labels.
    """
    # Generate random labels between 0 and num_classes - 1
    random_labels = [random.randint(0, num_classes - 1) for _ in range(num_images)]
    
    # Save labels to a text file
    with open(output_file, 'w') as f:
        for label in random_labels:
            f.write(f"{label}\n")

    print(f"Random labels generated and saved to {output_file}")

# Example usage:
noisy_dir = './random_greyscale_images/class1'
num_images = len(os.listdir(noisy_dir))  # Assuming you have 50 images
num_classes = 2  # Define the number of classes (you can adjust this)
output_file = 'fake_labels.txt'

generate_random_labels(num_images, num_classes, output_file)

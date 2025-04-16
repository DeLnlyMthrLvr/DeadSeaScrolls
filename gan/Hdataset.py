import os
from PIL import Image
import numpy as np

def load_hebrew_dataset(root_dir, target_size=(32, 32)):
    """
    Loads a Hebrew character dataset from a directory, resizes each image to target_size,
    and returns a single NumPy array that mimics the structure of a .npy file.
    
    Assumes that the root directory contains one folder per class and that each folder 
    contains .pgm images. All images are converted to grayscale.
    
    Args:
        root_dir (str): Path to the dataset root directory.
        target_size (tuple): Target size (width, height) to which each image will be resized.
        
    Returns:
        A NumPy array with shape (num_classes, num_samples, height, width) if all classes have
        the same number of images; otherwise, an object array.
    """
    raw_data = []
    # Get a sorted list of class directories
    classes = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )
    
    for cls in classes:
        class_path = os.path.join(root_dir, cls)
        images = []
        # Process each .pgm file in the class directory
        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith('.pgm'):
                img_path = os.path.join(class_path, fname)
                # Load and process the image: convert to grayscale, resize and convert to array.
                img = Image.open(img_path).convert('L')
                img = img.resize(target_size)
                img_array = np.array(img)  # shape: (32, 32)
                # Optionally, convert to float and normalize if your training pipeline expects this.
                img_array = img_array.astype(np.float32) / 255.0
                images.append(img_array)
        # Convert the list of images for this class into a NumPy array.
        images = np.array(images)
        raw_data.append(images)
    
    # Attempt to stack everything into a single NumPy array.
    # This will only work if each class has the same number of images.
    raw_data_array = np.array(raw_data, dtype=object)
    return raw_data_array
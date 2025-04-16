import os
from PIL import Image
import numpy as np

def load_the_dataset(root_dir, target_size=(32, 32)):
    
    raw_data = []
    # Get a sorted list of class directories
    classes = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )
    
    for cls in classes:
        class_path = os.path.join(root_dir, cls)
        images = []
        if class_path == 'data/alphabet/Kaf-final':
            continue
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
                images.append(np.transpose([img_array], (1, 2, 0)))  # Transpose to (height, width)
        # Convert the list of images for this class into a NumPy array.
        images = np.array(images)
        raw_data.append(images)
    
    # raw_data_array = np.array(raw_data)
    return raw_data
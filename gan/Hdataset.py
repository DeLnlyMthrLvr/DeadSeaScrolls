import os
from PIL import Image
import numpy as np
from alphabet import load_alphabet, char_token
from synthetic import create_alphabet_image
import numpy as np
from pathlib import Path
import tqdm as tqdm
from PIL import Image

def find_bbox(img):
    """
    img : 2D numpy array (grayscale) or 3D (H×W×C).
    Assumes the background value is the most frequent pixel in img.
    Returns (y0, y1, x0, x1) inclusive bounds.
    """
    # If color, collapse to gray‐scale or just look at one channel:
    if img.ndim == 3:
        arr = img[...,0]
    else:
        arr = img

    # find background = most common value
    vals, counts = np.unique(arr, return_counts=True)
    bg = vals[np.argmax(counts)]

    # mask of “foreground” pixels
    mask = (arr != bg)
    if not mask.any():
        # nothing but background
        return 0, img.shape[0]-1, 0, img.shape[1]-1

    # get coordinates of non‐bg pixels
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return y0, y1, x0, x1

def crop_to_bbox(img):
    y0, y1, x0, x1 = find_bbox(img)
    # for a 2D array:
    return img[y0:y1+1, x0:x1+1]

def load_the_dataset(root_dir, target_size=(32, 32), balance_classes=False, add_data=None):
    
    if balance_classes:
        raw_alphabet = load_alphabet()
        letters = list(char_token.keys())[:len(raw_alphabet)+1]	
        class_sizes = [len(raw_alphabet[cls]) for cls in raw_alphabet]
        if add_data is not None:
            add_data = np.max(class_sizes)
        reverse_class = [add_data-cs for cs in class_sizes]

        for i, ammount in enumerate(reverse_class):
            let = f'{letters[i]}'.split(".")[1].replace("_", "-")
            #print(let)
            if let == 'Kaf-final':
                let = 'Zayin'
                oi = i
                i = 26
            file_count = len(list(Path(f"data/alphabet/{let}").glob("*.pgm")))
            ammount = add_data - file_count
            print(f"Class {i} has {file_count}")
            if ammount > 0:
                print(f"Class {i} has {ammount} missing samples. {letters[i]}")
                for j in tqdm.tqdm(range(ammount)):
                    sample = create_alphabet_image([letters[i]], (200,200), raw_alphabet)
                    out = Image.fromarray(crop_to_bbox(sample.image))
                    out = out.resize((32, 32))
                    out.save(f"data/alphabet/{let}/syntetic_{j}.pgm")

         
            if let == 'Kaf-final':
                i = oi

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
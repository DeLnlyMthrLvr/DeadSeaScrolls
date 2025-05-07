import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from synthetic import DataGenerator, SynthSettings

def crop_to_content(pil_img, bg_thresh=250):
    """
    Crops away any border where all pixels are >= bg_thresh (nearly white).
    Returns the tightest bounding‐box around any “ink” (pixels < bg_thresh).
    """
    arr = np.array(pil_img)
    # mask of “ink” pixels
    mask = arr < bg_thresh
    coords = np.argwhere(mask)
    if coords.size == 0:
        # no ink found, return original
        return pil_img
    # coords[:,0] = y indices, coords[:,1] = x indices
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    # crop box: (left, upper, right, lower) — right/lower exclusive
    return pil_img.crop((x0, y0, x1+1, y1+1))

class UnpairedImageDataset(Dataset):
    """
    Dataset for unpaired images from two domains:
    - Domain A: synthetic scrolls generated once and cached as tensors
    - Domain B: real scroll images loaded once and cached as tensors
    Applies separate preprocessing (resize, inversion, normalization) per domain at init.
    """
    def __init__(
        self,
        synthetic_generator: DataGenerator,
        num_synthetic: int,
        root_real: str,
        img_size: tuple = (120, 300),
        transform_synth=None,
        transform_real=None
    ):
        # Define transforms
        self.transform_synth = transform_synth or transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            # Synthetic: already black-bg/white-text, normalize
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform_real = transform_real or transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            # Invert real domain: white-bg -> black-bg
            # transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Generate synthetic images once and transform to tensors
        tokens, _, scrolls, _ = synthetic_generator.generate_passages_scrolls(
            N=num_synthetic,
            skip_char_seg=True
        )
        self.synthetic_tensors = []
        for arr in scrolls:
            img = Image.fromarray(arr)
            tensor = self.transform_synth(img)
            self.synthetic_tensors.append(tensor)

        # Load real images once and transform to tensors
        real_files = sorted([
            os.path.join(root_real, f)
            for f in os.listdir(root_real)
            if f.lower().endswith(('zed.jpg'))
        ])
        self.real_tensors = []
        for p in real_files:
            img = Image.open(p).convert('L')
            img = crop_to_content(img, bg_thresh=250)
            tensor = self.transform_real(img)
            self.real_tensors.append(tensor)

    def __len__(self):
        # Cycle through the larger domain
        return max(len(self.synthetic_tensors), len(self.real_tensors))

    def __getitem__(self, idx):
        # Synthetic sample (cycled)
        img_A = self.synthetic_tensors[idx % len(self.synthetic_tensors)]
        # Real sample (random)
        img_B = random.choice(self.real_tensors)
        return img_A, img_B


def create_unpaired_dataloader(
    synthetic_generator: DataGenerator,
    num_synthetic: int,
    root_real: str,
    img_size: tuple = (120, 300),
    batch_size: int = 16,
    num_workers: int = 4
) -> DataLoader:
    dataset = UnpairedImageDataset(
        synthetic_generator=synthetic_generator,
        num_synthetic=num_synthetic,
        root_real=root_real,
        img_size=img_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
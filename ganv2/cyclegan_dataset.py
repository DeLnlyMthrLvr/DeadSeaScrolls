import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Assume DataGenerator and SynthSettings are available from your synthetic pipeline
from synthetic import DataGenerator, SynthSettings

class UnpairedImageDataset(Dataset):
    """
    Dataset for unpaired images from two domains:
    - Domain A: synthetic scrolls generated on-the-fly via DataGenerator
    - Domain B: real scroll images loaded from disk
    """
    def __init__(
        self,
        synthetic_generator: DataGenerator,
        num_synthetic: int,
        root_real: str,
        img_size: tuple = (120, 300),
        transform=None
    ):
        tokens, _, scrolls, _ = synthetic_generator.generate_passages_scrolls(
            N=num_synthetic,
            skip_char_seg=True
        )
        self.synthetic_images = [Image.fromarray((arr * 255).astype(np.uint8)) for arr in scrolls]
        self.real_paths = sorted([
            os.path.join(root_real, f)
            for f in os.listdir(root_real)
            if f.lower().endswith(('zed.jpg'))
        ])
        self.transform = transform or transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return max(len(self.synthetic_images), len(self.real_paths))

    def __getitem__(self, idx):
        img_A = self.synthetic_images[idx % len(self.synthetic_images)]
        path_B = random.choice(self.real_paths)
        img_B = Image.open(path_B).convert('L')
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
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

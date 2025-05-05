from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import torch
import numpy as np
import sys
import os
import torchvision.transforms.functional as F
import image_creator
import pandas as pd
from torchvision.io import read_image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import synthetic

class ScrollLineDataset(Dataset):
    def __init__(self, tokens_parquet, image_dir, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = pd.read_parquet(tokens_parquet, engine='pyarrow')
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.tokens.at[idx,"image"]) + ".png"
        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        image_tokens = self.tokens.at[idx, "tokens"]
        token_ids = self.tokenizer.add_control_tokens(image_tokens)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        return image, token_tensor


class ScrollLineDatasetIterable(IterableDataset):
    def __init__(self, tokenizer, image_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=1))


    def line_iterator(self):
        while True:
            tokens, seg, scrolls, lines = self.generator.generate_passages_scrolls(100, skip_char_seg=False)

            for i in range(scrolls.shape[0]):
                image_tokens = tokens[i]
                image_lines = synthetic.extract_lines_cc(scrolls[i], lines[i])
                n_indicies = min(len(image_tokens), len(image_lines))

                for idx in range(n_indicies):
                    image = Image.fromarray(image_lines[idx])
                    image = image_creator.pad(image, self.image_size)
                    image = self.transform(image)
                    token_ids = self.tokenizer.add_control_tokens(image_tokens[idx])
                    token_tensor = torch.tensor(token_ids, dtype=torch.long)
                    yield image, token_tensor

    def __iter__(self):
        return self.line_iterator()
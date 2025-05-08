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
from tokenizer import Tokenizer

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

class ScrollLineDatasetWithPadding(Dataset):
    def __init__(self, tokens_parquet, image_dir, tokenizer, image_size):
        self.tokenizer = tokenizer
        self.tokens = pd.read_parquet(tokens_parquet, engine='pyarrow')
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.tokens.at[idx,"image"]) + ".png"
        image = Image.open(image_path).convert("L")
        image = image_creator.pad(image, self.image_size)
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
            tokens, seg, scrolls, lines = self.generator.generate_ngram_scrolls(100, skip_char_seg=False)

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
    
class MixedDatasetIterable(IterableDataset):
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
            tokens_bible, seg_bible, scrolls_bible, lines_bible = self.generator.generate_passages_scrolls(100, skip_char_seg=True)
            tokens_grams, seg_grams, scrolls_grams, lines_grams = self.generator.generate_ngram_scrolls(100, skip_char_seg=True)
            #assert scrolls_bible.shape[0] == scrolls_grams.shape[0]
            for i in range(scrolls_grams.shape[0]):
                image_tokens = None
                scrolls = None
                lines = None
                for mode in range(0,2):
                    if mode == 0:
                        image_tokens = tokens_bible[i]
                        scrolls = scrolls_bible
                        lines = lines_bible
                    if mode == 1:
                        image_tokens = tokens_grams[i]
                        scrolls = scrolls_grams
                        lines = lines_grams

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
    


class ScrollLineNoiseDatasetIterable(IterableDataset):
    def __init__(self, 
                 tokenizer: Tokenizer, 
                 image_size: tuple, 
                 warp_strength: tuple[float, float] = (0, 10),
                 perlin_strength: tuple[float, float] = (0, 0.25),
                 cutout_size: tuple[int, int] = (0, 40),
                 n_noise_masks: int = 30,
                 n_images_per_level: int = 500,
                 n_progress: int = 5,
                 downscale: float = 0.5):
        
        super().__init__()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.n_noise_masks = n_noise_masks

        self.n_images_per_level = n_images_per_level
        self.alphabet = synthetic.load_alphabet()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.downscale = downscale
        #self.generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=downscale))
        self.perlin_strength = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
        self.warp_strength  = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
        self.cutout_size = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)
        normal = synthetic.SynthSettings(downscale_factor=downscale)
        self.noise = synthetic.Noise(normal.downscale_size)
        self.noise.create_masks(N=n_noise_masks)

    def line_iterator(self):
        while True:
            for level, (p, w, c) in enumerate(zip(self.perlin_strength, self.warp_strength, self.cutout_size, strict=True)):

                settings = synthetic.SynthSettings(
                    warp_noise=w > 0,
                    warp_noise_strength=w,
                    cutout_noise=c > 0,
                    cutout_noise_size=c,
                    downscale_factor=self.downscale
                )
                generator = synthetic.DataGenerator(settings, self.alphabet)

                tokens, seg, scrolls, lines = generator.generate_passages_scrolls(self.n_images_per_level, skip_char_seg=False)
                if p > 0:
                    modified_scrolls = self.noise.damage(scrolls, strength=p)
                for i in range(scrolls.shape[0]):
                    image_tokens = tokens[i]
                    image_lines = synthetic.extract_lines_cc(modified_scrolls[i], lines[i])
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
    

class NoiseMixedDatasetIterable(IterableDataset):
    def __init__(self, 
                 tokenizer: Tokenizer, 
                 image_size: tuple, 
                 warp_strength: tuple[float, float] = (0, 3),
                 perlin_strength: tuple[float, float] = (0.01, 0.1),
                 cutout_size: tuple[int, int] = (0, 20),
                 n_noise_masks: int = 30,
                 n_images_per_level: int = 500,
                 n_progress: int = 5,
                 downscale: float = 0.5):
        
        super().__init__()
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.n_noise_masks = n_noise_masks

        self.n_images_per_level = n_images_per_level
        self.alphabet = synthetic.load_alphabet()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.downscale = downscale
        #self.generator = synthetic.DataGenerator(settings=synthetic.SynthSettings(downscale_factor=downscale))
        self.perlin_strength = np.linspace(perlin_strength[0], perlin_strength[1], n_progress)
        self.warp_strength  = np.linspace(warp_strength[0], warp_strength[1], n_progress).round().astype(int)
        self.cutout_size = np.linspace(cutout_size[0], cutout_size[1], n_progress).round().astype(int)
        normal = synthetic.SynthSettings(downscale_factor=downscale)
        self.noise = synthetic.Noise(normal.downscale_size)
        self.noise.create_masks(N=n_noise_masks)

    def line_iterator(self):
        while True:
            for level, (p, w, c) in enumerate(zip(self.perlin_strength, self.warp_strength, self.cutout_size, strict=True)):

                settings = synthetic.SynthSettings(
                    warp_noise=w > 0,
                    warp_noise_strength=w,
                    cutout_noise=c > 0,
                    cutout_noise_size=c,
                    downscale_factor=self.downscale
                )
                generator = synthetic.DataGenerator(settings, self.alphabet)

                tokens_bible, seg_bible, scrolls_bible, lines_bible = generator.generate_passages_scrolls(100, skip_char_seg=True)
                tokens_grams, seg_grams, scrolls_grams, lines_grams = generator.generate_ngram_scrolls(100, skip_char_seg=True)

                if p > 0:
                    modified_scrolls_bible = self.noise.damage(scrolls_bible, strength=p)
                    modified_scrolls_grams = self.noise.damage(scrolls_grams, strength=p)
                for i in range(scrolls_bible.shape[0]):
                    image_tokens = None
                    modified_scrolls = None
                    lines = None
                    for mode in range(0,2):
                        if mode == 0:
                            image_tokens = tokens_bible[i]
                            modified_scrolls = modified_scrolls_bible
                            lines = lines_bible
                        if mode == 1:
                            image_tokens = tokens_grams[i]
                            modified_scrolls = modified_scrolls_grams
                            lines = lines_grams
                        #image_tokens = tokens[i]
                        image_lines = synthetic.extract_lines_cc(modified_scrolls[i], lines[i])
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